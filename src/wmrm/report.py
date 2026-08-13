"""Machine-readable result of a run, written to `--report FILE`.

Why a file rather than the exit status: see `errors.py`. The short version is that
exit 1 currently means six unrelated things, so a wrapper cannot decide from it
whether to retry, to stop, or to ask a human.

Two rules this module follows, because both have been got wrong before:

- **Every field comes from something that exists.** No summary number is invented for
  the report's benefit. `unblend` carries the four fields `Unblend` actually has;
  there is no "background floor", because that number is a hand-measured constant in
  the README, not a value the code computes.
- **Fields absent for an engine are `null`, not zero.** `unblend` is only populated by
  the un-blend engine and `propainter` only by ProPainter. A reader has to be able to
  tell "this engine does not report that" from "it reported zero".
"""

from __future__ import annotations

import json
import os
import platform
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import __version__
from .errors import outcome_for

SCHEMA = 1

#: Closed set. A consumer is entitled to switch on this exhaustively, so adding a
#: member is an interface change -- keep it in step with the service that reads it.
OUTCOMES = (
    "ok",
    "coverage_under",
    "coverage_inconclusive",
    "verify_failed",
    "input_error",
    "usage_error",
    "detect_failed",
    "encode_error",
    "engine_unavailable",
    "oom",
    "interrupted",
    "canceled",
    "internal",
    # Not produced by `wmrm run` -- the pod server adds it when the pixels were fine and
    # only publishing the result failed. Kept in this enum so both sides validate against
    # one list.
    "upload_failed",
)

#: Outcomes after which there is nothing to run again. `interrupted` is absent on
#: purpose: it means the process was stopped from outside, not that the work is wrong.
TERMINAL_FAILURES = (
    "coverage_under",
    "coverage_inconclusive",
    "verify_failed",
    "input_error",
    "usage_error",
    "detect_failed",
    "canceled",
)


@dataclass
class RunContext:
    """What `_log_config` worked out, kept instead of only printed.

    Before this, the engine label, the device label, the tile size and the probe
    numbers existed only as f-strings inside the banner. Anything that wanted them
    had to recompute them and risk disagreeing with what the banner said.
    """

    info: Any                      # probe.VideoInfo
    box: Any                       # region.Box
    tile: Any                      # region.Size
    engine: str                    # args.quality
    engine_label: str              # "unblend+lama", "propainter (flow propagation...)"
    device: str                    # requested: auto | cpu | cuda | mps
    device_label: str              # resolved, e.g. "cuda (NVIDIA A40, 44.3 GiB)"
    box_source: str = "given"      # given | detect | preset

    def to_dict(self) -> dict:
        return {
            "engine": self.engine,
            "engine_label": self.engine_label,
            "device": self.device,
            "device_label": self.device_label,
            "video": {
                "width": self.info.width,
                "height": self.info.height,
                "fps": float(self.info.fps),
                "frames": self.info.nframes or None,
                "duration_seconds": round(float(self.info.duration), 3),
                "has_audio": bool(self.info.has_audio),
                "pix_fmt": self.info.pix_fmt,
            },
            "box": {
                "x": self.box.x, "y": self.box.y, "w": self.box.w, "h": self.box.h,
                "source": self.box_source,
            },
            "tile": {"w": self.tile.w, "h": self.tile.h},
        }


def coverage_to_dict(cov) -> dict:
    """Serialize `coverage.Coverage`.

    `signal_gradient` and `signal_variance` are **booleans** -- which signal fired,
    not how strongly. They have been written up as floats before; a reader that
    expects a number gets `true` and silently mis-renders it.
    """
    return {
        "ok": bool(cov.ok),
        "inconclusive": bool(cov.inconclusive),
        "ring_px": int(cov.ring_px),
        "residual_fraction": float(cov.residual_fraction),
        "reach": {k: int(v) for k, v in cov.reach.items()},
        "suggested": (
            {"x": cov.suggested.x, "y": cov.suggested.y,
             "w": cov.suggested.w, "h": cov.suggested.h}
            if cov.suggested else None
        ),
        "signal_gradient": bool(cov.signal_gradient),
        "signal_variance": bool(cov.signal_variance),
    }


def unblend_to_dict(fitted) -> dict:
    """Serialize `unblend.Unblend` -- the four fields it has, and no others.

    `opaque_fraction` is the one to watch: a mark that is nearly opaque has little for
    un-blend to recover, and the run is closer to inpainting than to recovery. It is
    reported rather than turned into a failure, because choosing a different engine is
    a judgement about the footage and there is no measured threshold to make it here.
    """
    return {
        "residual_before": round(float(fitted.residual_before), 4),
        "residual": round(float(fitted.residual), 4),
        "alpha_scale": round(float(fitted.alpha_scale), 4),
        "opaque_fraction": round(float(fitted.opaque_fraction), 6),
    }


class ReportWriter:
    """Accumulates a run's facts, then writes exactly one JSON file.

    Written on **every** exit path, including failures -- a report that only appears
    when things go well is useless to the caller that has to decide what happened.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.started = time.time()
        self._d: dict[str, Any] = {
            "schema": SCHEMA,
            "wmrm_version": __version__,
            "ok": False,
            "outcome": None,
            "input": None,
            "output": None,
            "engine": None,
            "engine_label": None,
            "device": None,
            "device_label": None,
            "video": None,
            "box": None,
            "tile": None,
            "coverage": None,
            "unblend": None,
            "propainter": None,
            "verify": None,
            "elapsed_seconds": None,
            "host": platform.node(),
            "error": None,
        }
        self._written = False

    # -- collection ------------------------------------------------------------ #

    def set_paths(self, src: Path, dst: Path | None) -> None:
        self._d["input"] = str(src)
        self._d["output"] = str(dst) if dst else None

    def set_context(self, ctx: RunContext) -> None:
        self._d.update(ctx.to_dict())

    def set_coverage(self, cov) -> None:
        self._d["coverage"] = coverage_to_dict(cov)

    def set_unblend(self, fitted) -> None:
        self._d["unblend"] = unblend_to_dict(fitted)

    def set_propainter(self, **fields) -> None:
        cur = self._d.get("propainter") or {}
        cur.update({k: v for k, v in fields.items() if v is not None})
        self._d["propainter"] = cur

    def set_verify(self, result) -> None:
        self._d["verify"] = result.to_dict()

    # -- termination ----------------------------------------------------------- #

    def ok(self, dst: Path | None = None) -> None:
        if dst is not None:
            self._d["output"] = str(dst)
        self._finish("ok", ok=True)

    def fail_outcome(self, outcome: str, message: str) -> None:
        """Terminate with an outcome decided by the caller, not by an exception.

        Used where the failure is a verdict rather than a crash: verification came
        back not-ok, so there is no exception to map.
        """
        self._d["error"] = {"code": outcome, "message": message}
        self._d["output"] = None
        self._finish(outcome, ok=False)

    def fail(self, exc: BaseException) -> None:
        cancel = os.environ.get("WMRM_CANCEL_REQUESTED") == "1"
        outcome = outcome_for(exc, cancel_requested=cancel)
        message = str(exc) or type(exc).__name__
        err: dict[str, Any] = {"code": outcome, "message": message,
                               "exception": type(exc).__name__}
        if outcome == "internal":
            # Kept because an unmapped failure is the case where the message alone is
            # never enough. Truncated because this ends up in a database row.
            err["traceback"] = "".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__)
            )[-8000:]
        self._d["error"] = err
        # An output that was mid-write is not an output. Saying otherwise invites a
        # caller to upload a truncated file.
        if outcome != "ok":
            self._d["output"] = None
        self._finish(outcome, ok=False)

    def _finish(self, outcome: str, *, ok: bool) -> None:
        if self._written:
            return
        assert outcome in OUTCOMES, f"unknown outcome {outcome!r}"
        self._d["outcome"] = outcome
        self._d["ok"] = ok
        self._d["elapsed_seconds"] = round(time.time() - self.started, 3)
        self._write()

    def flush(self) -> None:
        """Last resort, for a path that returned without saying how.

        Idempotent: a run that already terminated normally leaves this a no-op. Without
        it, an early `return 0` somewhere would leave no file at all, and a missing file
        is indistinguishable to the caller from a process that died before starting.
        """
        if not self._written:
            self._finish("internal", ok=False)

    def _write(self) -> None:
        self._written = True
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # Atomic: a reader polling this path must never see half a document. The
            # service does exactly that.
            tmp = self.path.with_name(f".{self.path.name}.partial")
            tmp.write_text(json.dumps(self._d, indent=2, sort_keys=False) + "\n")
            os.replace(tmp, self.path)
        except OSError as exc:                       # pragma: no cover
            # Never let reporting break the run it is reporting on.
            import sys
            print(f"[wmrm] note: could not write report {self.path}: {exc}",
                  file=sys.stderr)
