"""Typed failures, so a caller can tell them apart.

Exit codes cannot carry this information and never could. `main()` returns 2 for
every known exception, 3 for an unexpected one and 130 for Ctrl-C, while fourteen
`raise SystemExit("error: ...")` sites return 1 -- and 1 today means all of "the
input does not exist", "ffmpeg failed", "--box and --preset were both passed" and
"verification failed". A wrapper reading only the exit status cannot decide whether
to retry, to stop, or to ask a human, because those three answers all arrive as 1.

So the outcome travels in the report file (`--report`, see `report.py`) and this
module is what fills it in. Each class carries the `outcome` string it maps to, which
keeps the mapping next to the failure instead of in a table somewhere else that drifts.

Deliberately narrow: only the failures reachable from `wmrm run` are modelled, because
`run` is the command the service wraps. `detect`, `batch`, `grid` and `pull` keep their
`SystemExit`s -- converting them buys nothing today and would be a bigger diff to review.
"""

from __future__ import annotations


class WmrmError(Exception):
    """Base for failures we can name. `outcome` is what the report records."""

    outcome = "internal"


class UsageError(WmrmError):
    """The command was asked for something contradictory or incomplete.

    Not retryable, and worth distinguishing from every other failure: it means the
    *caller* is wrong, so retrying on another machine will fail identically. When the
    caller is an API this is a bug in the API, not a problem with the video.
    """

    outcome = "usage_error"


class InputMissing(WmrmError):
    """The input file is not there."""

    outcome = "input_error"


class OutputLocked(WmrmError):
    """Another run holds the lock on this output path (see `lock.py`)."""

    outcome = "internal"


class CoverageUnder(WmrmError):
    """The box provably does not cover the whole mark.

    A hard stop rather than a warning. The alternative is shipping a video with a
    sliver of watermark still in it, which is the failure this project exists to
    prevent and the one nobody notices until after delivery.
    """

    outcome = "coverage_under"


class CoverageInconclusive(WmrmError):
    """Coverage could not answer, so a human has to look.

    Distinct from `CoverageUnder` because the box may well be fine: the ring around it
    looks mark-like everywhere, which happens when the background is itself static
    (fixed camera on a plain surface). No statistic separates mark from wall there.
    """

    outcome = "coverage_inconclusive"


def outcome_for(exc: BaseException, *, cancel_requested: bool = False) -> str:
    """Map an exception to a report outcome.

    `cancel_requested` distinguishes the two ways a run stops early, which the process
    cannot tell apart on its own -- SIGTERM looks identical whether it came from an
    operator cancelling the job or from the pod being restarted underneath it. The
    caller sets `WMRM_CANCEL_REQUESTED=1` before signalling when it is a cancellation;
    everything else is `interrupted` and is safe to run again.
    """
    from .detect import DetectError
    from .pipeline import EncodeError
    from .probe import ProbeError, ToolMissing

    if isinstance(exc, WmrmError):
        return exc.outcome
    if isinstance(exc, (KeyboardInterrupt, SystemExit)) and cancel_requested:
        return "canceled"
    if isinstance(exc, KeyboardInterrupt):
        return "interrupted"
    if isinstance(exc, ProbeError):
        return "input_error"
    if isinstance(exc, DetectError):
        return "detect_failed"
    if isinstance(exc, EncodeError):
        return "encode_error"
    if isinstance(exc, ToolMissing):
        return "engine_unavailable"

    # ProPainter and torch are optional imports: a machine with neither installed must
    # still be able to map every other failure, so these are matched by name rather
    # than imported. Checking the MRO by name also catches subclasses.
    names = {cls.__name__ for cls in type(exc).__mro__}
    if "ProPainterError" in names:
        return "engine_unavailable"
    if "OutOfMemoryError" in names or "CudaOutOfMemoryError" in names:
        return "oom"

    if isinstance(exc, SystemExit):
        # A site that has not been converted yet. Reported as `internal` with the
        # message kept verbatim, so it shows up as something to fix rather than being
        # silently absorbed into a generic failure.
        return "internal"
    return "internal"
