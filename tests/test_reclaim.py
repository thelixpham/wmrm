#!/usr/bin/env python
"""Disk reclamation tests. Run directly: `python tests/test_reclaim.py`.

No GPU, no network, no real video -- every "output" here is a file of zeros, because what
is being tested is which files survive, not what is in them.

The cases are chosen to be the ones where getting it wrong destroys something real: an
output that is not in R2 yet, an input that lives outside the job directory because an
operator pointed at it, and an interrupted job whose parts are the only reason a retry is
cheap. The happy path -- a delivered job's 26 GB going away -- is the easy half.
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        PASSED.append(name)
        print(f"  PASS  {name}" + (f" -- {detail}" if detail else ""))
    else:
        FAILED.append(name)
        print(f"  FAIL  {name}" + (f" -- {detail}" if detail else ""))


def make_cfg(tmp: Path, *, retention_hours: float = 48.0, min_free_gb: float = 0.01):
    from wmrm.server.config import Config

    os.environ.update({
        "WMRM_POD_ID": "testpod",
        "WMRM_POD_TOKEN": "dev-token",
        "WMRM_WORK_DIR": str(tmp / "work"),
        "WMRM_STATE": str(tmp / "state"),
        "WMRM_RETENTION_HOURS": str(retention_hours),
        "WMRM_MIN_FREE_GB": str(min_free_gb),
    })
    os.environ.pop("WMRM_RECLAIM", None)
    cfg = Config.from_env()
    cfg.ensure_dirs()
    return cfg


def write(path: Path, size: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(b"\0" * size)
    return path


def make_job(cfg, store, job_id: str, *, state: str, output_key: str | None = None,
             parts: bool = False, age: float = 0.0):
    """A job directory that looks like one a real run left behind."""
    job_dir = cfg.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    write(job_dir / "source.mp4", 4096)                 # the download
    write(job_dir / "source-clean.mp4", 2048)           # the output
    write(job_dir / "report.json", 64)                  # records, must survive
    write(job_dir / "source-preview.png", 128)
    write(job_dir / "source-preset.json", 32)
    if parts:
        write(job_dir / "source-clean.mp4.parts" / "part-0000.mp4", 1024)
        write(job_dir / "source-clean.mp4.parts" / "manifest.json", 32)

    rec = store.create(job_id=job_id, spec={"engine": "unblend"}, work_dir=str(job_dir))
    rec.set_state(state, outcome="ok" if state == "succeeded" else state,
                  outputKey=output_key)
    if age:
        rec.set(finishedAt=time.time() - age)
    return rec, job_dir


def names(job_dir: Path) -> set[str]:
    return {p.name for p in job_dir.iterdir()} if job_dir.is_dir() else set()


# --------------------------------------------------------------------------- #

def test_after_delivery(tmp: Path) -> None:
    print("\n[after a delivery]")
    from wmrm.server import reclaim
    from wmrm.server.store import JobStore

    cfg = make_cfg(tmp)
    store = JobStore(cfg.state_dir)

    rec, job_dir = make_job(cfg, store, "job_delivered", state="succeeded",
                            output_key="output/job_delivered/source-clean.mp4")
    freed = reclaim.after_delivery(cfg, rec)
    left = names(job_dir)

    check("the source is gone", "source.mp4" not in left)
    check("the output is gone", "source-clean.mp4" not in left)
    check("report.json stays", "report.json" in left)
    check("the preview stays", "source-preview.png" in left)
    check("the preset stays", "source-preset.json" in left)
    check("it reports what it freed", freed == 4096 + 2048, f"{freed} bytes")
    check("the record says so", bool(rec.data.get("reclaimedAt")),
          f"reason={rec.data.get('reclaimReason')}")
    check("the job is still readable through the API shape",
          rec.public()["reclaimedAt"] is not None
          and rec.public()["outputKey"].endswith("source-clean.mp4"))

    # Idempotent: a second pass has nothing to do and must not undo the record.
    again = reclaim.after_delivery(cfg, rec)
    check("a second pass frees nothing", again == 0, f"{again} bytes")


def test_undelivered_is_untouched(tmp: Path) -> None:
    print("\n[nothing in R2 means nothing to delete]")
    from wmrm.server import reclaim
    from wmrm.server.store import JobStore

    cfg = make_cfg(tmp)
    store = JobStore(cfg.state_dir)

    # `succeeded` with no outputKey is the local-output case: this file is the only copy.
    rec, job_dir = make_job(cfg, store, "job_local_out", state="succeeded")
    freed = reclaim.after_delivery(cfg, rec)
    check("a succeeded job with no outputKey keeps its output", freed == 0
          and "source-clean.mp4" in names(job_dir))

    # A failed upload is the case the whole `upload_failed` outcome exists for: the file
    # passed verification and only delivery went wrong, so it must survive to be retried.
    rec2, dir2 = make_job(cfg, store, "job_upload_failed", state="failed")
    rec2.set(outcome="upload_failed")
    freed2 = reclaim.after_delivery(cfg, rec2)
    check("an upload_failed job keeps its output", freed2 == 0
          and "source-clean.mp4" in names(dir2))


def test_live_job_is_untouched(tmp: Path) -> None:
    print("\n[a running job]")
    from wmrm.server import reclaim
    from wmrm.server.store import JobStore

    cfg = make_cfg(tmp, retention_hours=0.0)
    store = JobStore(cfg.state_dir)
    rec, job_dir = make_job(cfg, store, "job_running", state="running", age=99999)

    summary = reclaim.sweep(cfg, store)
    check("a running job's files survive a zero-retention sweep",
          "source.mp4" in names(job_dir), f"freed={summary['freedBytes']}")
    check("and it is not counted as kept-for-later either", summary["kept"] == 0)


def test_retention_window(tmp: Path) -> None:
    print("\n[the retention window]")
    from wmrm.server import reclaim
    from wmrm.server.store import JobStore

    cfg = make_cfg(tmp, retention_hours=48.0)
    store = JobStore(cfg.state_dir)

    young, young_dir = make_job(cfg, store, "job_young", state="interrupted",
                                parts=True, age=3600)
    old, old_dir = make_job(cfg, store, "job_old", state="interrupted",
                            parts=True, age=72 * 3600)
    done, done_dir = make_job(cfg, store, "job_done", state="succeeded",
                              output_key="output/job_done/source-clean.mp4", age=60)

    summary = reclaim.sweep(cfg, store)

    check("an interrupted job inside the window keeps its parts",
          "source-clean.mp4.parts" in names(young_dir))
    check("and is reported as kept", summary["kept"] == 1, str(summary["kept"]))
    check("past the window the parts go", "source-clean.mp4.parts" not in names(old_dir))
    check("a delivered job does not wait for the window",
          "source-clean.mp4" not in names(done_dir))
    check("both are named in the summary",
          set(summary["jobs"]) == {"job_old", "job_done"}, str(summary["jobs"]))
    check("records survive everywhere",
          all("report.json" in names(d) for d in (young_dir, old_dir, done_dir)))


def test_orphan_dirs(tmp: Path) -> None:
    print("\n[directories no job claims]")
    from wmrm.server import reclaim
    from wmrm.server.store import JobStore

    cfg = make_cfg(tmp)
    store = JobStore(cfg.state_dir)

    stale = cfg.work_dir / "job_no_record"
    write(stale / "source.mp4", 4096)
    old = time.time() - 2 * reclaim.ORPHAN_DIR_MIN_AGE
    os.utime(stale, (old, old))

    fresh = cfg.work_dir / "job_just_made"
    write(fresh / "source.mp4", 4096)

    summary = reclaim.sweep(cfg, store)
    check("an old directory with no job record is removed", not stale.exists(),
          str(summary["orphanDirs"]))
    check("a fresh one is left alone -- it may be a job being submitted right now",
          fresh.is_dir())


def test_containment(tmp: Path) -> None:
    print("\n[containment]")
    from wmrm.server import reclaim
    from wmrm.server.store import JobStore

    cfg = make_cfg(tmp)
    store = JobStore(cfg.state_dir)

    # An operator's own file, outside the work dir -- the `input.kind: "local"` case.
    outside = write(tmp / "operators-own" / "keep-me.mp4", 4096)

    rec = store.create(job_id="job_evil", spec={},
                       work_dir=str(tmp / "operators-own"))
    rec.set_state("succeeded", outcome="ok", outputKey="output/x.mp4")
    freed = reclaim.after_delivery(cfg, rec)
    check("a workDir outside WMRM_WORK_DIR is refused", freed == 0 and outside.is_file())

    # The work dir itself, which would take every other job with it.
    rec2 = store.create(job_id="job_root", spec={}, work_dir=str(cfg.work_dir))
    rec2.set_state("succeeded", outcome="ok", outputKey="output/y.mp4")
    other = write(cfg.work_dir / "job_bystander" / "source.mp4", 1024)
    freed2 = reclaim.after_delivery(cfg, rec2)
    check("the work dir itself is refused", freed2 == 0 and other.is_file())


def test_low_space(tmp: Path) -> None:
    print("\n[under the free-space floor]")
    from wmrm.server import reclaim
    from wmrm.server.store import JobStore

    cfg = make_cfg(tmp, retention_hours=48.0, min_free_gb=100.0)
    store = JobStore(cfg.state_dir)

    # Everything here is inside its retention window, so only the floor can move it.
    interrupted, int_dir = make_job(cfg, store, "job_i", state="interrupted",
                                    parts=True, age=60)
    failed, failed_dir = make_job(cfg, store, "job_f", state="failed", age=120)

    order: list[str] = []
    real_reclaim = reclaim.reclaim

    def spy(cfg_, rec_, *, why):
        order.append(rec_.job_id)
        return real_reclaim(cfg_, rec_, why=why)

    # The floor is the trigger, and a test cannot fill a disk: `free_gb` is replaced so
    # the pod believes it is out of space, and then believes it is not.
    freed_at = [0]

    def fake_free_gb(_path):
        # Under the floor until something has been freed, then comfortably over it.
        return 1.0 if not order else 500.0

    reclaim.free_gb = fake_free_gb          # type: ignore[assignment]
    reclaim.reclaim = spy                   # type: ignore[assignment]
    try:
        summary = reclaim.sweep(cfg, store)
    finally:
        reclaim.reclaim = real_reclaim      # type: ignore[assignment]
        from wmrm.server.probe import free_gb as real_free
        reclaim.free_gb = real_free         # type: ignore[assignment]

    check("the floor overrides the window", summary["emergency"] == ["job_f"],
          str(summary["emergency"]))
    check("the failed job went first -- its parts are worth nothing",
          order and order[0] == "job_f", str(order))
    check("the interrupted job's parts were spared once there was room",
          "source-clean.mp4.parts" in names(int_dir))
    check("and the failed job's files are gone",
          "source.mp4" not in names(failed_dir))


def test_switch_off(tmp: Path) -> None:
    print("\n[WMRM_RECLAIM=off]")
    from wmrm.server import reclaim
    from wmrm.server.store import JobStore

    cfg = make_cfg(tmp, retention_hours=0.0)
    store = JobStore(cfg.state_dir)
    rec, job_dir = make_job(cfg, store, "job_off", state="succeeded",
                            output_key="output/job_off/source-clean.mp4", age=99999)
    os.environ["WMRM_RECLAIM"] = "off"
    try:
        check("enabled() is false", not reclaim.enabled())
        check("after_delivery does nothing", reclaim.after_delivery(cfg, rec) == 0)
        summary = reclaim.sweep(cfg, store)
        check("the sweep does nothing", summary["freedBytes"] == 0)
        check("the files are all still there", "source.mp4" in names(job_dir))
    finally:
        os.environ.pop("WMRM_RECLAIM", None)


def test_big_log_is_not_a_record(tmp: Path) -> None:
    print("\n[what counts as a record]")
    from wmrm.server import reclaim
    from wmrm.server.store import JobStore

    cfg = make_cfg(tmp)
    store = JobStore(cfg.state_dir)
    rec, job_dir = make_job(cfg, store, "job_logs", state="succeeded",
                            output_key="output/job_logs/source-clean.mp4")
    write(job_dir / "run.log", 4096)                                  # a real log
    write(job_dir / "huge.log", reclaim.KEEP_MAX_BYTES + 1)           # not evidence

    reclaim.after_delivery(cfg, rec)
    left = names(job_dir)
    check("a small log is kept", "run.log" in left)
    check("a log past the cap is not", "huge.log" not in left)


def test_submit_sweeps_before_refusing(tmp: Path) -> None:
    """A 507 is not allowed while the pod is sitting on space it does not need.

    The floor is what makes this reachable at all, and a test cannot fill a disk -- so
    `free_gb` as the route sees it is replaced: under the floor on the first look, over it
    once the sweep has run. The assertion is the status code the request gets *after* the
    disk gate (400 for a path that does not exist), because reaching that at all means the
    gate let it through.
    """
    print("\n[submit sweeps before it refuses]")
    from fastapi.testclient import TestClient

    from wmrm.server import app as app_module
    from wmrm.server.app import create_app
    from wmrm.server.config import Config
    from wmrm.server.store import JobStore

    cfg = make_cfg(tmp, min_free_gb=100.0)
    store = JobStore(cfg.state_dir)
    _rec, job_dir = make_job(cfg, store, "job_paid", state="succeeded",
                             output_key="output/job_paid/source-clean.mp4")
    os.environ["WMRM_LOCAL_INPUT_ROOT"] = str(tmp)
    cfg = Config.from_env()

    looks = [1.0, 500.0]                       # first look, then after the sweep
    real_free_gb = app_module.free_gb
    app_module.free_gb = lambda _p: looks.pop(0) if looks else 500.0  # type: ignore

    try:
        with TestClient(create_app(cfg)) as client:
            res = client.post(
                "/jobs",
                json={"schema": 1, "jobId": "job_new", "dispatchToken": "dt_x",
                      "input": {"kind": "local", "path": str(tmp / "nope.mp4")},
                      "output": {"kind": "local", "path": str(tmp / "out.mp4")},
                      "engine": "unblend",
                      "box": {"x": 1, "y": 1, "w": 8, "h": 8}},
                headers={"Authorization": "Bearer dev-token"})
            check("the disk gate was passed, not refused with 507",
                  res.status_code == 400, f"{res.status_code}: {res.text[:120]}")
            check("and the space came from the delivered job",
                  "source.mp4" not in names(job_dir), str(sorted(names(job_dir))))
    finally:
        app_module.free_gb = real_free_gb       # type: ignore
        os.environ.pop("WMRM_LOCAL_INPUT_ROOT", None)


def test_end_to_end_through_the_api(tmp: Path) -> None:
    """A real `wmrm run`, then the reclaim the runner does after a delivery.

    The upload itself is not exercised -- that needs R2 -- so the delivery is simulated by
    setting `outputKey`, which is exactly what `push_r2` returning does. What this covers
    that the unit tests do not is that a job's directory after a genuine run contains only
    things this code classifies correctly.
    """
    print("\n[end to end]")
    from fastapi.testclient import TestClient

    from wmrm.server import reclaim
    from wmrm.server.app import create_app

    fixture = Path(__file__).resolve().parent / "fixtures" / "detail-marked.mp4"
    if not fixture.is_file():
        print("  SKIP  no fixture; run tests/make_fixtures.py")
        return

    cfg = make_cfg(tmp)
    src = tmp / "src.mp4"
    src.write_bytes(fixture.read_bytes())
    out = cfg.job_dir("job_e2e") / "src-clean.mp4"
    os.environ["WMRM_LOCAL_INPUT_ROOT"] = str(tmp)
    from wmrm.server.config import Config
    cfg = Config.from_env()

    with TestClient(create_app(cfg)) as client:
        body = {
            "schema": 1, "jobId": "job_e2e", "dispatchToken": "dt_x",
            "input": {"kind": "local", "path": str(src)},
            "output": {"kind": "local", "path": str(out)},
            "engine": "unblend", "box": {"x": 8, "y": 8, "w": 40, "h": 16},
            "options": {"verify": False},
        }
        res = client.post("/jobs", json=body,
                          headers={"Authorization": "Bearer dev-token"})
        check("the job was accepted", res.status_code == 202, str(res.status_code))
        deadline = time.time() + 180
        state = "?"
        while time.time() < deadline:
            got = client.get("/jobs/job_e2e",
                             headers={"Authorization": "Bearer dev-token"}).json()
            state = got["state"]
            if state in ("succeeded", "failed", "needs_review", "canceled",
                         "interrupted"):
                break
            time.sleep(1.0)
        check("it finished", state == "succeeded", state)

        health = client.get("/health",
                            headers={"Authorization": "Bearer dev-token"}).json()
        check("/health reports what the pod is holding",
              isinstance(health["disk"]["heldGb"], float)
              and health["disk"]["reclaim"] is True,
              str(health["disk"]))

        # A local output is not in R2, so nothing may be deleted yet.
        rec = client.app.state.store.get("job_e2e")
        check("the output survives while it is the only copy",
              reclaim.after_delivery(cfg, rec) == 0 and out.is_file())

        # Now it is delivered, which is the only thing that changes.
        rec.set(outputKey="output/job_e2e/src-clean.mp4")
        freed = reclaim.after_delivery(cfg, rec)
        job_dir = cfg.job_dir("job_e2e")
        check("a delivered run's video files go", freed > 0
              and not any(p.suffix == ".mp4" for p in job_dir.iterdir()),
              f"{freed} bytes, left={sorted(names(job_dir))}")
        check("its report stays", (job_dir / "report.json").is_file())
    os.environ.pop("WMRM_LOCAL_INPUT_ROOT", None)


def main() -> int:
    tests = [
        test_after_delivery,
        test_undelivered_is_untouched,
        test_live_job_is_untouched,
        test_retention_window,
        test_orphan_dirs,
        test_containment,
        test_low_space,
        test_switch_off,
        test_big_log_is_not_a_record,
        test_submit_sweeps_before_refusing,
        test_end_to_end_through_the_api,
    ]
    for fn in tests:
        with tempfile.TemporaryDirectory(prefix="wmrm-reclaim-") as raw:
            fn(Path(raw))

    print(f"\n{len(PASSED)} passed, {len(FAILED)} failed")
    for name in FAILED:
        print(f"  FAILED: {name}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
