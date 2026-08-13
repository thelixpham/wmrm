"""Wire schemas, and the one piece of real logic that belongs with them:
turning a JSON options object into `wmrm run` arguments.

That translation is not mechanical, because several CLI flags exist only in the negative
(`--no-fp16`, `--pp-no-black-cuts`, `--no-resume`, `--no-verify`). Sending `"fp16": true`
must add nothing, and `false` must add the flag. Doing this in one place, with the
mapping written down, is the difference between that and a polarity bug that only shows
up as slightly wrong pixels.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from .config import ENGINES

Engine = Literal["unblend", "video", "high", "fast", "draft"]
CoverageGate = Literal["strict", "warn", "off"]

#: JSON name -> CLI flag, for options that take a value.
_VALUE_FLAGS: dict[str, str] = {
    "corner": "--corner",
    "samples": "--samples",
    "roiFrac": "--roi-frac",
    "gradThreshold": "--grad-threshold",
    "persistence": "--persistence",
    "maxArea": "--max-area",
    "dilate": "--dilate",
    "feather": "--feather",
    "margin": "--margin",
    "crf": "--crf",
    "x264Preset": "--x264-preset",
    "patchHold": "--patch-hold",
    "cacheTolerance": "--cache-tolerance",
    "unblendSamples": "--unblend-samples",
    "threads": "--threads",
    "device": "--device",
    "ppSegment": "--pp-segment",
    "ppPart": "--pp-part",
    "ppSubvideo": "--pp-subvideo",
    "ppWorkers": "--pp-workers",
    "raftIter": "--raft-iter",
    "ppMinShot": "--pp-min-shot",
    "ppSceneThreshold": "--pp-scene-threshold",
    "propainter": "--propainter",
    "coverageGate": "--coverage-gate",
}

#: JSON name -> flag added when the value is **false**. These are the inverted ones.
_NEGATIVE_FLAGS: dict[str, str] = {
    "fp16": "--no-fp16",
    "ppBlackCuts": "--pp-no-black-cuts",
    "resume": "--no-resume",
    "verify": "--no-verify",
}


class Box(BaseModel):
    x: int = Field(ge=0)
    y: int = Field(ge=0)
    w: int = Field(gt=0)
    h: int = Field(gt=0)

    def as_arg(self) -> str:
        return f"{self.x},{self.y},{self.w},{self.h}"


class InputSpec(BaseModel):
    """Where the source comes from.

    `r2` is the normal case: the pod holds R2 credentials and fetches the key itself
    with `wmrm.r2.download`, which is 8 parallel ranged GETs into a preallocated file
    with chunk-level resume. That matters at these sizes -- a single-stream download of
    100 GB is slower by the width of the parallelism, and a transfer that can only
    start from zero is one that may never finish.

    `url` (a presigned GET) stays because it is the variant that needs no credentials
    on the pod at all, and `local` covers a source already staged on the volume.
    """

    kind: Literal["r2", "url", "local"]
    key: str | None = None            # r2
    bucket: str | None = None         # r2, optional -- defaults to R2_BUCKET
    url: str | None = None            # url
    path: str | None = None           # local
    sizeBytes: int | None = None
    filename: str | None = None

    def model_post_init(self, _ctx: Any) -> None:
        if self.kind == "r2" and not self.key:
            raise ValueError("input.kind='r2' needs input.key")
        if self.kind == "url" and not self.url:
            raise ValueError("input.kind='url' needs input.url")
        if self.kind == "local" and not self.path:
            raise ValueError("input.kind='local' needs input.path")


class OutputSpec(BaseModel):
    """Where the result goes.

    `r2` means the pod uploads it itself, multipart and resumable, and reports the key.
    That removes an entire round of presigned-URL plumbing from the control plane: it
    does not have to mint part URLs, track an upload id, or expose routes for the pod
    to call back into.
    """

    kind: Literal["r2", "local"]
    key: str | None = None
    bucket: str | None = None
    path: str | None = None

    def model_post_init(self, _ctx: Any) -> None:
        if self.kind == "r2" and not self.key:
            raise ValueError("output.kind='r2' needs output.key")
        if self.kind == "local" and not self.path:
            raise ValueError("output.kind='local' needs output.path")


class JobSpec(BaseModel):
    schema_: int = Field(default=1, alias="schema")
    jobId: str = Field(min_length=1, max_length=128)
    dispatchToken: str = Field(min_length=1, max_length=256)
    callbackBaseUrl: str | None = None
    input: InputSpec
    output: OutputSpec
    engine: Engine
    box: Box | None = None
    options: dict[str, Any] = Field(default_factory=dict)
    heartbeatEverySeconds: int = Field(default=30, ge=5, le=3600)

    model_config = {"populate_by_name": True}

    @field_validator("jobId")
    @classmethod
    def _safe_job_id(cls, v: str) -> str:
        # This becomes a directory name and part of a log path. Anything that could climb
        # out of the work directory has no business being in it.
        if not all(c.isalnum() or c in "-_" for c in v):
            raise ValueError("jobId may contain only letters, digits, '-' and '_'")
        return v

    def argv(self, *, input_path: str, output_path: str, report_path: str) -> list[str]:
        """The exact `wmrm run` invocation, as a list -- never a shell string.

        Note what is absent: `--progress`. That flag does not exist; progress is on
        unless `--quiet` is passed. Building the command from a table rather than a
        format string is what stops that class of mistake reaching a nine-hour run.
        """
        if self.engine not in ENGINES:            # pragma: no cover -- Literal guards it
            raise ValueError(f"unknown engine {self.engine}")

        argv = ["wmrm", "run", input_path, "-o", output_path,
                "--report", report_path, "--quality", self.engine]
        if self.box is not None:
            argv += ["--box", self.box.as_arg()]

        opts = dict(self.options)
        # A gate is always stated. The CLI default is `warn` because the coverage check
        # false-positives on this project's own fixture, but an unattended run is exactly
        # the case that must not ship a maybe -- so absent means strict here, not warn.
        opts.setdefault("coverageGate", "strict")

        for name, value in opts.items():
            if value is None:
                continue
            if name in _NEGATIVE_FLAGS:
                if value is False:
                    argv.append(_NEGATIVE_FLAGS[name])
                continue
            flag = _VALUE_FLAGS.get(name)
            if flag is None:
                # Unknown keys are dropped rather than passed through. Forwarding them
                # would let a caller inject arbitrary argv.
                continue
            if isinstance(value, bool):
                # A value flag given a boolean is a caller mistake; guessing which way
                # they meant it is worse than ignoring it.
                continue
            argv += [flag, str(value)]
        return argv

    def gate(self) -> str:
        return str(self.options.get("coverageGate") or "strict")


class SubmitAccepted(BaseModel):
    accepted: bool = True
    jobId: str
    state: str
    workDir: str


class CancelAccepted(BaseModel):
    accepted: bool = True
    jobId: str
    state: str
    purge: bool = False


# --------------------------------------------------------------------------- #
# response shapes
#
# Declared so they reach the OpenAPI schema. A client generated from a spec whose
# only models are `ValidationError` has to guess the field names of everything it
# actually reads, which is the drift that generating a spec was meant to prevent.
# --------------------------------------------------------------------------- #

class Gpu(BaseModel):
    cuda: bool
    name: str | None = None
    vramTotalMb: int | None = None
    vramFreeMb: int | None = None
    #: Compute capabilities this torch build has kernels for. Worth reading before
    #: dispatching: a cu124 wheel on a Blackwell card installs cleanly, reports CUDA,
    #: loads the models, then dies at the first kernel launch.
    archList: list[str] = Field(default_factory=list)
    torch: str | None = None
    error: str | None = None


class Capacity(BaseModel):
    maxConcurrent: int
    running: int


class Disk(BaseModel):
    workDirPath: str
    workDirFreeGb: float
    minFreeGb: float


class R2Status(BaseModel):
    """Whether this pod can fetch and publish objects itself.

    `configured` false means `input.kind='r2'` and `output.kind='r2'` are refused with
    400; only `local` and `url` will work.
    """

    configured: bool
    bucket: str | None = None
    workers: int = 8
    reason: str | None = None


class Health(BaseModel):
    ok: bool
    schema_: int = Field(default=1, alias="schema")
    wmrmVersion: str
    podId: str
    gpu: Gpu
    ffmpeg: bool
    ffprobe: bool
    nvdec: bool
    #: Engines this machine can actually run -- GPU engines are omitted without CUDA.
    engines: list[str]
    capacity: Capacity
    disk: Disk
    r2: R2Status
    currentJobIds: list[str]
    uptimeSeconds: int

    model_config = {"populate_by_name": True}


class JobStatus(BaseModel):
    jobId: str
    #: preparing | downloading | detecting | running | uploading | succeeded | failed |
    #: needs_review | canceled | interrupted
    state: str
    phase: str | None = None
    engine: str | None = None
    outputKey: str | None = None
    box: dict[str, Any] | None = None
    progress: dict[str, Any] | None = None
    #: The report's outcome. `interrupted` means run it again; `canceled` means a person
    #: stopped it; `upload_failed` means the file is good and only delivery failed.
    outcome: str | None = None
    error: dict[str, Any] | None = None
    report: dict[str, Any] | None = None
    createdAt: float | None = None
    updatedAt: float | None = None
    startedAt: float | None = None
    finishedAt: float | None = None


class JobList(BaseModel):
    jobs: list[JobStatus]
    podId: str
