"""Wire schemas, and the one piece of real logic that belongs with them:
turning a JSON options object into `wmrm run` arguments.

That translation is not mechanical, because several CLI flags exist only in the negative
(`--no-fp16`, `--pp-no-black-cuts`, `--no-resume`, `--no-verify`). Sending `"fp16": true`
must add nothing, and `false` must add the flag. Doing this in one place, with the
mapping written down, is the difference between that and a polarity bug that only shows
up as slightly wrong pixels.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Union

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


# --------------------------------------------------------------------------- #
# input and output
#
# One model per kind, joined by a discriminator, rather than one model with every
# field of every kind left optional. The flat version validated the same requests,
# but it documented them wrongly: the generated schema listed key, bucket, url, path,
# sizeBytes and filename together, so the example payload showed six fields where a
# real request has one or two, and nothing said which belonged to which kind.
#
# Discriminating also moves the "kind=r2 needs key" rules out of a validator and into
# the types, where the error message names the field instead of the combination.
#
# All of them forbid extra keys. Pydantic ignores unknown fields by default, and with a
# discriminated union that is a trap: `{"kind": "r2", "path": "/etc/passwd"}` would
# validate, drop the path, and fetch from R2 -- so a caller who picked the wrong kind gets
# the other kind's behaviour with no complaint. The cost is forward compatibility: a newer
# control plane that adds a field breaks against an older pod. That is the right way round
# here, because it breaks loudly and `schema` exists to negotiate the change deliberately.
# --------------------------------------------------------------------------- #

class Strict(BaseModel):
    model_config = {"extra": "forbid"}


class InputR2(Strict):
    """The normal case: the pod fetches the key itself.

    `wmrm.r2.download` is 8 parallel ranged GETs into a preallocated file, resumable at
    chunk granularity. That matters at these sizes -- one connection is slower by the
    width of the parallelism, and a transfer that can only start from zero is one that
    may never finish.
    """

    kind: Literal["r2"] = "r2"
    key: str = Field(min_length=1, examples=["uploads/3d809a59-.../4K_MOGI-130.mp4"])
    #: Defaults to the pod's R2_BUCKET.
    bucket: str | None = None


class InputUrl(Strict):
    """A presigned GET, for a pod that holds no credentials.

    One connection rather than eight, and the URL can expire mid-transfer, so this is
    the fallback rather than the default.
    """

    kind: Literal["url"] = "url"
    url: str = Field(min_length=1)
    #: Lets the pod check free space before the first byte instead of at 90 GB.
    sizeBytes: int | None = Field(default=None, ge=0)
    filename: str | None = None


class InputLocal(Strict):
    """Already on the volume. Must resolve inside WMRM_LOCAL_INPUT_ROOT."""

    kind: Literal["local"] = "local"
    path: str = Field(min_length=1)


InputSpec = Annotated[
    Union[InputR2, InputUrl, InputLocal], Field(discriminator="kind")
]


class OutputR2(Strict):
    """The pod uploads the result itself and reports the key.

    This removes a whole round of presigned-URL plumbing from the control plane: no part
    URLs to mint, no upload id to track, no routes for the pod to call back into.
    """

    kind: Literal["r2"] = "r2"
    key: str = Field(min_length=1, examples=["output/job_01J.../4K_MOGI-130-clean.mp4"])
    bucket: str | None = None


class OutputLocal(Strict):
    """Leave it on disk. For trying things out without a bucket."""

    kind: Literal["local"] = "local"
    path: str = Field(min_length=1)


OutputSpec = Annotated[Union[OutputR2, OutputLocal], Field(discriminator="kind")]


class JobSpec(BaseModel):
    schema_: int = Field(default=1, alias="schema")
    jobId: str = Field(min_length=1, max_length=128)
    dispatchToken: str = Field(min_length=1, max_length=256)
    callbackBaseUrl: str | None = None
    input: InputSpec
    output: OutputSpec
    engine: Engine
    box: Box | None = None
    #: Any run flag, camelCased. The four that exist only in the negative on the CLI --
    #: fp16, ppBlackCuts, resume, verify -- are sent as positives and inverted here.
    options: dict[str, Any] = Field(
        default_factory=dict,
        examples=[{"device": "cuda", "coverageGate": "strict"}],
    )
    heartbeatEverySeconds: int = Field(default=30, ge=5, le=3600)

    model_config = {
        "populate_by_name": True,
        # A worked example, because the generated one is assembled field by field and ends
        # up showing every optional key at once -- which reads as "all of this is required".
        # A real request is this short.
        "json_schema_extra": {
            "examples": [
                {
                    "schema": 1,
                    "jobId": "job_01JBQ7Z8K3M4N5P6Q7R8S9T0V1",
                    "dispatchToken": "dt_9f3c2a...",
                    "callbackBaseUrl": "https://wmrm.example.com",
                    "input": {
                        "kind": "r2",
                        "key": "uploads/3d809a59-.../4K_MOGI-130.mp4",
                    },
                    "output": {
                        "kind": "r2",
                        "key": "output/job_01JBQ7Z8K3M4N5P6Q7R8S9T0V1/"
                               "4K_MOGI-130-clean.mp4",
                    },
                    "engine": "video",
                    "box": {"x": 1640, "y": 20, "w": 205, "h": 62},
                    "options": {"device": "cuda"},
                }
            ]
        },
    }

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
