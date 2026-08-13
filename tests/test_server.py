#!/usr/bin/env python
"""Pod API tests. Run directly: `python tests/test_server.py`.

No GPU and no network needed. The one test that runs a real job uses `--quality unblend`
on the small fixture, because the point is the wrapper's behaviour, not the engine's.

Covers the things that are cheap to get wrong and expensive to discover in production:
auth, capacity refusal, path traversal, idempotent re-dispatch, argv translation of the
inverted flags, and that cancelling actually kills the whole process group rather than
leaving ffmpeg behind.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

HERE = Path(__file__).resolve().parent
FIXTURE = HERE / "fixtures" / "detail-marked.mp4"

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        PASSED.append(name)
        print(f"  PASS  {name}" + (f" -- {detail}" if detail else ""))
    else:
        FAILED.append(name)
        print(f"  FAIL  {name}" + (f" -- {detail}" if detail else ""))


def make_client(tmp: Path, *, token: str = "dev-token", max_concurrent: int = 1,
                min_free_gb: float = 0.01, input_root: Path | None = None):
    """Build a client. **Always use it as a context manager.**

    Two things depend on that and both fail silently otherwise: `startup` (and therefore
    orphan adoption and the machine probe) only runs inside the `with`, and outside it
    every request gets a fresh event loop, so a background job is abandoned the moment
    its request finishes. The symptom of getting this wrong is a submit that appears to
    block for the whole length of the job.
    """
    from fastapi.testclient import TestClient

    from wmrm.server.app import create_app
    from wmrm.server.config import Config

    os.environ.update({
        "WMRM_POD_ID": "testpod",
        "WMRM_WORK_DIR": str(tmp / "work"),
        "WMRM_STATE": str(tmp / "state"),
        "WMRM_MAX_CONCURRENT": str(max_concurrent),
        "WMRM_MIN_FREE_GB": str(min_free_gb),
    })
    if token:
        os.environ["WMRM_POD_TOKEN"] = token
    else:
        os.environ.pop("WMRM_POD_TOKEN", None)
    if input_root:
        os.environ["WMRM_LOCAL_INPUT_ROOT"] = str(input_root)
    else:
        os.environ.pop("WMRM_LOCAL_INPUT_ROOT", None)

    cfg = Config.from_env()
    return TestClient(create_app(cfg)), cfg


def job_body(job_id: str, *, src: Path, out: Path, engine: str = "unblend",
             **extra) -> dict:
    body = {
        "schema": 1,
        "jobId": job_id,
        "dispatchToken": "dt_test",
        "input": {"kind": "local", "path": str(src)},
        "output": {"kind": "local", "path": str(out)},
        "engine": engine,
        "box": {"x": 379, "y": 427, "w": 91, "h": 43},
        "options": {"device": "cpu", "coverageGate": "off"},
        "heartbeatEverySeconds": 5,
    }
    body.update(extra)
    return body


# --------------------------------------------------------------------------- #

def test_auth(tmp: Path) -> None:
    print("\n[auth]")
    client, _ = make_client(tmp / "auth")
    with client as c:
        check("/live needs no token", c.get("/live").status_code == 200)
        check("/live leaks nothing", c.get("/live").content == b"")
        check("/health without a token is 401",
              c.get("/health").status_code == 401)
        check("/health with a wrong token is 401",
              c.get("/health", headers={"authorization": "Bearer nope"}).status_code == 401)
        r = c.get("/health", headers={"authorization": "Bearer dev-token"})
        check("/health with the right token is 200", r.status_code == 200)

    # A pod with no token configured refuses rather than serving openly: an unconfigured
    # deploy must not look like a working one when the API can start GPU jobs.
    client2, _ = make_client(tmp / "auth-none", token="")
    with client2 as c2:
        check("a pod with no token configured refuses with 503",
              c2.get("/health").status_code == 503)


def test_health_shape(tmp: Path) -> None:
    print("\n[health]")
    client, _ = make_client(tmp / "health")
    with client as c:
        h = c.get("/health", headers={"authorization": "Bearer dev-token"}).json()
    for key in ("ok", "schema", "wmrmVersion", "podId", "gpu", "engines",
                "capacity", "disk", "currentJobIds", "uptimeSeconds"):
        check(f"health has {key}", key in h)
    check("podId comes from the environment", h["podId"] == "testpod", h["podId"])
    check("work dir is namespaced by podId",
          h["disk"]["workDirPath"].endswith("testpod"), h["disk"]["workDirPath"])
    check("gpu.archList is present", isinstance(h["gpu"]["archList"], list))

    cuda = h["gpu"]["cuda"]
    if cuda:
        check("engines include video when CUDA is present", "video" in h["engines"])
    else:
        # Reporting an engine this machine cannot run invites a dispatch that starts and
        # then takes days: ProPainter on CPU is 0.27 fps.
        check("engines exclude video with no CUDA", "video" not in h["engines"],
              str(h["engines"]))
        check("engines still include unblend", "unblend" in h["engines"])


def test_validation(tmp: Path) -> None:
    print("\n[validation]")
    root = FIXTURE.parent
    auth = {"authorization": "Bearer dev-token"}
    out = tmp / "valid" / "out.mp4"

    client, _ = make_client(tmp / "valid", input_root=root)
    with client as c:
        r = c.post("/jobs", headers=auth, json={"jobId": "nope"})
        check("a malformed body is 400", r.status_code == 400, str(r.status_code))

        r = c.post("/jobs", headers=auth, json=job_body("bad-id!", src=FIXTURE, out=out))
        check("a jobId with unsafe characters is 400", r.status_code == 400)

        # Resolved before checking, so `..` and symlinks are both caught.
        r = c.post("/jobs", headers=auth, json=job_body(
            "traversal", src=root / ".." / ".." / "etc" / "passwd", out=out))
        check("a path outside the input root is 400", r.status_code == 400,
              str(r.json().get("detail"))[:80])

        r = c.post("/jobs", headers=auth, json=job_body(
            "missing", src=root / "does-not-exist.mp4", out=out))
        check("a missing input path is 400", r.status_code == 400)

    # Without WMRM_LOCAL_INPUT_ROOT set, a local path has no root to be inside.
    client2, _ = make_client(tmp / "valid-noroot")
    with client2 as c2:
        r = c2.post("/jobs", headers=auth, json=job_body("noroot", src=FIXTURE, out=out))
        check("a local input with no configured root is 400", r.status_code == 400)

    client3, _ = make_client(tmp / "valid-disk", input_root=root,
                             min_free_gb=10_000_000)
    with client3 as c3:
        r = c3.post("/jobs", headers=auth, json=job_body("nodisk", src=FIXTURE, out=out))
        check("not enough disk is 507", r.status_code == 507, str(r.status_code))


def test_argv_translation() -> None:
    print("\n[argv]")
    from wmrm.server.models import JobSpec

    spec = JobSpec.model_validate(job_body("argv", src=FIXTURE, out=Path("/tmp/o.mp4")))
    argv = spec.argv(input_path="IN", output_path="OUT", report_path="R")

    check("argv starts with wmrm run", argv[:2] == ["wmrm", "run"])
    check("--report is passed", "--report" in argv)
    check("--quality is passed", argv[argv.index("--quality") + 1] == "unblend")
    check("--box is formatted x,y,w,h", "379,427,91,43" in argv)
    # The flag does not exist; progress is on unless --quiet is given.
    check("--progress is never emitted", "--progress" not in argv)

    # Inverted flags: true must add nothing, false must add the negative flag.
    spec_t = JobSpec.model_validate(job_body(
        "argvt", src=FIXTURE, out=Path("/tmp/o.mp4"),
        options={"fp16": True, "ppBlackCuts": True, "resume": True, "verify": True}))
    a_t = spec_t.argv(input_path="IN", output_path="OUT", report_path="R")
    check("fp16=true adds no flag", "--no-fp16" not in a_t)
    check("ppBlackCuts=true adds no flag", "--pp-no-black-cuts" not in a_t)
    check("resume=true adds no flag", "--no-resume" not in a_t)
    check("verify=true adds no flag", "--no-verify" not in a_t)

    spec_f = JobSpec.model_validate(job_body(
        "argvf", src=FIXTURE, out=Path("/tmp/o.mp4"),
        options={"fp16": False, "ppBlackCuts": False, "resume": False, "verify": False}))
    a_f = spec_f.argv(input_path="IN", output_path="OUT", report_path="R")
    check("fp16=false adds --no-fp16", "--no-fp16" in a_f)
    check("ppBlackCuts=false adds --pp-no-black-cuts", "--pp-no-black-cuts" in a_f)
    check("resume=false adds --no-resume", "--no-resume" in a_f)
    check("verify=false adds --no-verify", "--no-verify" in a_f)

    # Absent gate means strict for an unattended run, even though the CLI default is warn.
    spec_g = JobSpec.model_validate(job_body(
        "argvg", src=FIXTURE, out=Path("/tmp/o.mp4"), options={}))
    a_g = spec_g.argv(input_path="IN", output_path="OUT", report_path="R")
    check("an absent coverageGate becomes strict",
          a_g[a_g.index("--coverage-gate") + 1] == "strict")

    # Every pixel-deciding flag must survive translation, including the three that were
    # missing from an earlier version of this mapping.
    spec_p = JobSpec.model_validate(job_body(
        "argvp", src=FIXTURE, out=Path("/tmp/o.mp4"),
        options={"x264Preset": "slow", "cacheTolerance": 2.5,
                 "ppSceneThreshold": 0.4, "crf": 20}))
    a_p = spec_p.argv(input_path="IN", output_path="OUT", report_path="R")
    for flag, value in (("--x264-preset", "slow"), ("--cache-tolerance", "2.5"),
                        ("--pp-scene-threshold", "0.4"), ("--crf", "20")):
        check(f"{flag} is translated",
              flag in a_p and a_p[a_p.index(flag) + 1] == value)

    # Unknown keys are dropped, not forwarded: forwarding would be argv injection.
    spec_x = JobSpec.model_validate(job_body(
        "argvx", src=FIXTURE, out=Path("/tmp/o.mp4"),
        options={"--rm": "-rf", "totallyUnknown": "x"}))
    a_x = spec_x.argv(input_path="IN", output_path="OUT", report_path="R")
    check("unknown option keys are dropped", "-rf" not in a_x and "--rm" not in a_x)


def test_capacity_and_idempotency(tmp: Path) -> None:
    print("\n[capacity]")
    root = FIXTURE.parent
    base = tmp / "cap"
    client, _ = make_client(base, input_root=root, max_concurrent=1)
    auth = {"authorization": "Bearer dev-token"}

    with client as c:
        t0 = time.monotonic()
        r1 = c.post("/jobs", headers=auth,
                    json=job_body("j1", src=FIXTURE, out=base / "o1.mp4"))
        took = time.monotonic() - t0
        check("the first job is accepted with 202", r1.status_code == 202,
              str(r1.status_code))
        # The run itself takes ~78s on this fixture. Anything near that means the work
        # happened inside the request, which the RunPod proxy would cut off at 100s.
        check("submit returns promptly rather than waiting for the run", took < 5.0,
              f"{took:.2f}s")

        r2 = c.post("/jobs", headers=auth,
                    json=job_body("j2", src=FIXTURE, out=base / "o2.mp4"))
        check("a second job on a busy pod is 409", r2.status_code == 409,
              str(r2.status_code))

        # A re-dispatch whose response was lost must not start the job twice.
        r3 = c.post("/jobs", headers=auth,
                    json=job_body("j1", src=FIXTURE, out=base / "o1.mp4"))
        check("re-submitting a known jobId is 200, not a second run",
              r3.status_code == 200 and r3.json()["accepted"] is False,
              str(r3.status_code))

        r = c.get("/jobs/j1", headers=auth)
        check("GET /jobs/{id} works", r.status_code == 200)
        check("state is a live one", r.json()["state"] in
              ("preparing", "downloading", "running"), r.json()["state"])
        check("GET /jobs/{id} does not hand back the spec",
              "spec" not in r.json())
        check("GET /jobs/{unknown} is 404",
              c.get("/jobs/nope", headers=auth).status_code == 404)
        check("GET /jobs lists it", "j1" in
              [j["jobId"] for j in c.get("/jobs", headers=auth).json()["jobs"]])

        print("\n[cancel]")
        for _ in range(40):            # wait for the child to actually be up
            if subprocess.run(["pgrep", "-f", "wmrm run"],
                              capture_output=True).returncode == 0:
                break
            time.sleep(0.5)
        pgrep_before = subprocess.run(["pgrep", "-f", "wmrm run"], capture_output=True)
        check("the child process is running", pgrep_before.returncode == 0)

        r = c.post("/jobs/j1/cancel", headers=auth, json={})
        check("cancel is accepted with 202", r.status_code == 202, str(r.status_code))

        for _ in range(80):
            state = c.get("/jobs/j1", headers=auth).json()["state"]
            if state in ("canceled", "failed", "succeeded", "interrupted"):
                break
            time.sleep(0.5)
        final = c.get("/jobs/j1", headers=auth).json()
        # `canceled`, not `interrupted`: the child cannot tell who signalled it, so the
        # pod -- which knows it asked -- decides.
        check("a cancelled job ends up canceled, not interrupted",
              final["state"] == "canceled",
              f"state={final['state']} outcome={final['outcome']}")

        r = c.get("/jobs/j1/log", headers=auth, params={"tail": 20})
        check("the log endpoint returns text", r.status_code == 200)

        # DELETE refuses while live and works once it is not.
        check("DELETE on a finished job is 204",
              c.delete("/jobs/j1", headers=auth).status_code == 204)
        check("the job is gone afterwards",
              c.get("/jobs/j1", headers=auth).status_code == 404)

    # The whole process group has to go. `wmrm run` spawns two ffmpeg children: a decoder
    # holding the input and an encoder holding the output. Checked after the client is
    # closed so nothing of ours is still starting things.
    time.sleep(2)
    leftover_py = subprocess.run(["pgrep", "-f", "wmrm run"], capture_output=True)
    check("no wmrm run process is left behind", leftover_py.returncode != 0,
          leftover_py.stdout.decode()[:60])
    enc = subprocess.run(["pgrep", "-f", "ffmpeg.*o1.mp4"], capture_output=True)
    check("no ffmpeg encoder is left behind", enc.returncode != 0,
          enc.stdout.decode()[:60])
    dec = subprocess.run(["pgrep", "-f", "ffmpeg.*detail-marked"], capture_output=True)
    check("no ffmpeg decoder is left behind", dec.returncode != 0,
          dec.stdout.decode()[:60])


def test_r2_input_accepted(tmp: Path) -> None:
    """A `kind: "r2"` job must be accepted on a pod that has credentials.

    The gap this closes: `resolve_input` returned early only for `kind == "url"`, so when
    the `r2` kind was added every r2 job fell into the local-path branch and was refused
    for not having WMRM_LOCAL_INPUT_ROOT -- an error about a field the caller never sent.
    Nothing caught it, because the r2 tests all used a pod *without* credentials, which is
    refused earlier for a different reason.

    Fake credentials are enough here: `r2_configured` only checks that the four variables
    are present, and the transfer that would really contact R2 happens in the background
    after the 202 this asserts on.
    """
    print("\n[r2 input accepted]")
    auth = {"authorization": "Bearer dev-token"}

    saved = {k: os.environ.get(k) for k in
             ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET")}
    os.environ.update(
        R2_ACCOUNT_ID="0" * 32,
        R2_ACCESS_KEY_ID="test-key",
        R2_SECRET_ACCESS_KEY="test-secret",
        R2_BUCKET="remove-watermark",
    )
    try:
        client, cfg = make_client(tmp / "r2-in")
        check("the pod reports R2 configured", cfg.r2_configured is True,
              str(cfg.r2_reason()))

        with client as c:
            h = c.get("/health", headers=auth).json()
            check("/health agrees", h["r2"]["configured"] is True)
            check("/health names the bucket",
                  h["r2"]["bucket"] == "remove-watermark", str(h["r2"]["bucket"]))

            # The exact minimal shape the queue sends: no output, no box, no options.
            r = c.post("/jobs", headers=auth, json={
                "jobId": "r2_in",
                "dispatchToken": "dt_x",
                "callbackBaseUrl": "https://example.invalid",
                "input": {"kind": "r2", "key": "uploads/abc/clip.mp4"},
                "engine": "unblend",
            })
            check("an r2 input is accepted with 202", r.status_code == 202,
                  f"{r.status_code} {str(r.json())[:160]}")
            check("the refusal is not about WMRM_LOCAL_INPUT_ROOT",
                  "WMRM_LOCAL_INPUT_ROOT" not in str(r.json()),
                  str(r.json())[:120])

            c.post("/jobs/r2_in/cancel", headers=auth, json={})

        # And the unit that broke: r2 and url must not be treated as local paths.
        from wmrm.server.config import Config
        from wmrm.server.models import JobSpec
        from wmrm.server.runner import JobRunner
        from wmrm.server.store import JobStore

        cfg2 = Config.from_env()
        runner = JobRunner(cfg2, JobStore(cfg2.state_dir))
        for kind, extra in (
            ("r2", {"key": "uploads/a/b.mp4"}),
            ("url", {"url": "https://example.invalid/a.mp4"}),
        ):
            spec = JobSpec.model_validate({
                "jobId": f"k_{kind}", "dispatchToken": "dt",
                "input": {"kind": kind, **extra}, "engine": "unblend",
            })
            check(f"resolve_input returns None for kind={kind}",
                  runner.resolve_input(spec) is None)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_output_omitted(tmp: Path) -> None:
    """An absent `output` must be accepted, and must not be read as a demand for R2.

    This is the shape the queue actually sends: the pod derives
    `output/<jobId>/<stem>-clean<ext>` because it already knows both halves. An earlier
    version required the field, and when it was made optional the credential check still
    read `spec.output.kind` -- so a payload without it crashed with AttributeError instead
    of being accepted. Hence a test for the omission specifically.
    """
    print("\n[output omitted]")
    root = FIXTURE.parent
    auth = {"authorization": "Bearer dev-token"}

    saved = {k: os.environ.pop(k, None) for k in
             ("R2_ACCOUNT_ID", "R2_ENDPOINT", "R2_ACCESS_KEY_ID",
              "R2_SECRET_ACCESS_KEY", "R2_BUCKET", "AWS_ACCESS_KEY_ID",
              "AWS_SECRET_ACCESS_KEY", "S3_BUCKET", "CLOUDFLARE_ACCOUNT_ID")}
    try:
        client, cfg = make_client(tmp / "no-output", input_root=root)
        check("this pod has no R2, so a derived output stays local",
              cfg.r2_configured is False)
        with client as c:
            r = c.post("/jobs", headers=auth, json={
                "jobId": "no_output",
                "dispatchToken": "dt",
                "input": {"kind": "local", "path": str(FIXTURE)},
                "engine": "unblend",
                "options": {"device": "cpu", "coverageGate": "off"},
            })
            check("a job without output is accepted", r.status_code == 202,
                  f"{r.status_code} {str(r.json())[:120]}")

            # The derived plan is recorded before the run finishes, so the caller can see
            # where the result is going rather than waiting to find out.
            for _ in range(40):
                body = c.get("/jobs/no_output", headers=auth).json()
                if body.get("state") in ("running", "succeeded", "failed"):
                    break
                time.sleep(0.5)
            check("the pod planned an output path itself",
                  bool(body.get("state")), str(body.get("state")))
            c.post("/jobs/no_output/cancel", headers=auth, json={})
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v

    # Minimal payload: four fields and nothing else.
    from wmrm.server.models import JobSpec

    spec = JobSpec.model_validate({
        "jobId": "minimal",
        "dispatchToken": "dt",
        "input": {"kind": "r2", "key": "uploads/a/b.mp4"},
        "engine": "video",
    })
    check("four fields validate", spec.jobId == "minimal")
    check("output defaults to None", spec.output is None)
    check("box defaults to None", spec.box is None)
    check("schema defaults to 1", spec.schema_ == 1)
    check("heartbeat defaults to 30", spec.heartbeatEverySeconds == 30)


def test_r2_kinds(tmp: Path) -> None:
    """A pod without credentials must refuse `kind: "r2"` at submit time.

    Accepting it and failing later moves a clear 400 onto someone else's machine as a
    mysterious job failure -- and the control plane has already spent a dispatch on it.
    """
    print("\n[r2 kinds]")
    auth = {"authorization": "Bearer dev-token"}

    # Make sure this process looks credential-less regardless of what ran before.
    saved = {k: os.environ.pop(k, None) for k in
             ("R2_ACCOUNT_ID", "R2_ENDPOINT", "R2_ACCESS_KEY_ID",
              "R2_SECRET_ACCESS_KEY", "R2_BUCKET", "AWS_ACCESS_KEY_ID",
              "AWS_SECRET_ACCESS_KEY", "S3_BUCKET", "CLOUDFLARE_ACCOUNT_ID")}
    try:
        client, cfg = make_client(tmp / "r2-off")
        check("a pod with no credentials reports r2 unconfigured",
              cfg.r2_configured is False)
        check("and says why", bool(cfg.r2_reason()), str(cfg.r2_reason())[:60])

        with client as c:
            h = c.get("/health", headers=auth).json()
            check("/health reports the r2 block", "r2" in h)
            check("/health says r2 is not configured",
                  h["r2"]["configured"] is False)

            r = c.post("/jobs", headers=auth, json={
                "schema": 1, "jobId": "r2in", "dispatchToken": "dt",
                "input": {"kind": "r2", "key": "uploads/a/b.mp4"},
                "output": {"kind": "local", "path": str(tmp / "o.mp4")},
                "engine": "unblend", "options": {"device": "cpu"},
            })
            check("input.kind=r2 without credentials is 400", r.status_code == 400,
                  str(r.status_code))
            check("the refusal explains itself",
                  "R2 credentials" in str(r.json().get("detail")),
                  str(r.json().get("detail"))[:70])

            r = c.post("/jobs", headers=auth, json={
                "schema": 1, "jobId": "r2out", "dispatchToken": "dt",
                "input": {"kind": "local", "path": str(FIXTURE)},
                "output": {"kind": "r2", "key": "output/x/y.mp4"},
                "engine": "unblend", "options": {"device": "cpu"},
            })
            check("output.kind=r2 without credentials is 400", r.status_code == 400,
                  str(r.status_code))
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v

    # Shape validation happens before any credential check, so it holds either way.
    #
    # InputSpec/OutputSpec are discriminated unions, not classes, so they are validated
    # through a TypeAdapter. That is the point of the union: each kind requires only its
    # own fields, and the schema says so instead of listing every field of every kind.
    from pydantic import TypeAdapter, ValidationError

    from wmrm.server.models import InputR2, InputSpec, OutputSpec

    inputs = TypeAdapter(InputSpec)
    outputs = TypeAdapter(OutputSpec)
    for adapter, kwargs, label in (
        (inputs, {"kind": "r2"}, "input.kind=r2 without key"),
        (inputs, {"kind": "url"}, "input.kind=url without url"),
        (inputs, {"kind": "local"}, "input.kind=local without path"),
        (inputs, {"kind": "nonsense", "key": "x"}, "an unknown input kind"),
        (outputs, {"kind": "r2"}, "output.kind=r2 without key"),
        (outputs, {"kind": "local"}, "output.kind=local without path"),
    ):
        try:
            adapter.validate_python(kwargs)
            check(f"{label} is rejected", False, "no error")
        except ValidationError:
            check(f"{label} is rejected", True)

    ok = inputs.validate_python({"kind": "r2", "key": "uploads/a/b.mp4"})
    check("an r2 input resolves to InputR2", isinstance(ok, InputR2))
    check("a bucket override is optional", ok.bucket is None)
    check("the key survives", ok.key == "uploads/a/b.mp4")

    # The variants are genuinely separate: a field from another kind is not accepted just
    # because it exists somewhere in the union.
    try:
        inputs.validate_python({"kind": "r2", "key": "a/b.mp4", "path": "/etc/passwd"})
        check("a field from another kind is rejected", False, "no error")
    except ValidationError:
        check("a field from another kind is rejected", True)


def test_orphan_adoption(tmp: Path) -> None:
    """A job left running by a previous process is published as interrupted.

    The alternative is a job that stays `running` forever: it holds a slot on this
    machine and the control plane never learns to retry it.
    """
    print("\n[orphans]")
    from wmrm.server.store import JobStore

    state = tmp / "orphan" / "state" / "testpod"
    store = JobStore(state)
    rec = store.create(job_id="stale", spec={"engine": "unblend"},
                       work_dir=str(tmp / "orphan" / "work"))
    rec.set_state("running")
    check("the stale job starts out live", store.get("stale").is_live)

    client, _ = make_client(tmp / "orphan")
    # Inside the `with`, so the startup hook that does the adopting actually runs.
    with client as c:
        r = c.get("/jobs/stale", headers={"authorization": "Bearer dev-token"})
    check("startup adopted it", r.status_code == 200)
    body = r.json()
    check("it is now interrupted, not canceled",
          body["state"] == "interrupted" and body["outcome"] == "interrupted",
          f"state={body['state']} outcome={body['outcome']}")


def test_hmac() -> None:
    print("\n[webhook signing]")
    from wmrm.server.hooks import sign

    body = b'{"a":1}'
    a = sign("secret", 1700000000, body)
    b = sign("secret", 1700000000, body)
    check("signing is deterministic", a == b)
    check("a different timestamp changes the signature",
          sign("secret", 1700000001, body) != a)
    check("a different body changes the signature",
          sign("secret", 1700000000, b'{"a":2}') != a)
    check("a different secret changes the signature",
          sign("other", 1700000000, body) != a)

    # The timestamp is inside the signed string, so it cannot be edited on its own.
    import hashlib
    import hmac as _hmac
    expect = _hmac.new(b"secret", b"v1:1700000000:" + body, hashlib.sha256).hexdigest()
    check("the signed string is v1:{ts}:{body}", a == expect)


def main() -> int:
    if not FIXTURE.is_file():
        print(f"SKIP: {FIXTURE} missing -- run: python tests/make_fixtures.py")
        return 0
    try:
        import fastapi  # noqa: F401
        from fastapi.testclient import TestClient  # noqa: F401
    except ImportError:
        print("SKIP: the 'serve' extra is not installed "
              "-- run: uv pip install -e '.[serve]'")
        return 0

    tmp = Path(tempfile.mkdtemp(prefix="wmrm-server-test-"))
    try:
        test_hmac()
        test_argv_translation()
        test_auth(tmp)
        test_health_shape(tmp)
        test_validation(tmp)
        test_r2_input_accepted(tmp)
        test_output_omitted(tmp)
        test_r2_kinds(tmp)
        test_orphan_adoption(tmp)
        test_capacity_and_idempotency(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n{len(PASSED)} passed, {len(FAILED)} failed")
    if FAILED:
        for name in FAILED:
            print(f"  FAILED: {name}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
