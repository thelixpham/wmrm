#!/usr/bin/env python
"""Report and outcome-mapping tests. Run directly: `python tests/test_report.py`.

Fast and offline -- no video is processed here. The end-to-end checks that a real run
writes the right outcome live in the acceptance commands in the README; this file pins
the things that are easy to break silently:

- the outcome enum is closed and every mapping lands inside it
- a report is written on **every** exit path, including ones that raise
- `unblend` carries the four fields the fitted object has and no invented fifth
- `coverage` signals are booleans, which they have been written up as floats before
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    (PASSED if condition else FAILED).append(name)
    print(f"  {'PASS' if condition else 'FAIL'}  {name}" + (f" -- {detail}" if detail else ""))


def test_outcome_mapping() -> None:
    print("\n[outcome mapping]")
    from wmrm.detect import DetectError
    from wmrm.errors import (CoverageInconclusive, CoverageUnder, InputMissing,
                             OutputLocked, UsageError, outcome_for)
    from wmrm.pipeline import EncodeError
    from wmrm.probe import ProbeError, ToolMissing
    from wmrm.report import OUTCOMES

    cases = [
        (UsageError("x"), "usage_error"),
        (InputMissing("x"), "input_error"),
        (CoverageUnder("x"), "coverage_under"),
        (CoverageInconclusive("x"), "coverage_inconclusive"),
        (OutputLocked("x"), "internal"),
        (ProbeError("x"), "input_error"),
        (DetectError("x"), "detect_failed"),
        (EncodeError("x"), "encode_error"),
        (ToolMissing("x"), "engine_unavailable"),
        (KeyboardInterrupt(), "interrupted"),
        (ValueError("x"), "internal"),
        (RuntimeError("x"), "internal"),
    ]
    for exc, expected in cases:
        got = outcome_for(exc)
        check(f"{type(exc).__name__} -> {expected}", got == expected, got)

    # The process cannot tell a cancellation from a restart, so the caller says which.
    check("KeyboardInterrupt with cancel_requested -> canceled",
          outcome_for(KeyboardInterrupt(), cancel_requested=True) == "canceled")
    check("SystemExit with cancel_requested -> canceled",
          outcome_for(SystemExit("x"), cancel_requested=True) == "canceled")
    # An unconverted SystemExit must surface as something to fix, not vanish.
    check("a bare SystemExit -> internal", outcome_for(SystemExit("x")) == "internal")

    # Anything a mapping can produce has to be in the enum, or a consumer switching on
    # it exhaustively has a hole.
    for exc, expected in cases:
        check(f"{expected} is a declared outcome", expected in OUTCOMES)


def test_report_written_on_every_path() -> None:
    print("\n[report writing]")
    from wmrm.errors import CoverageUnder
    from wmrm.report import ReportWriter

    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "ok.json"
        w = ReportWriter(p)
        w.set_paths(Path("/in.mp4"), Path("/out.mp4"))
        w.ok(dst=Path("/out.mp4"))
        d = json.loads(p.read_text())
        check("a successful run writes ok=true", d["ok"] is True and d["outcome"] == "ok")
        check("it records the output path", d["output"] == "/out.mp4")
        check("elapsed_seconds is filled in", isinstance(d["elapsed_seconds"], float))

        p = Path(tmp) / "fail.json"
        w = ReportWriter(p)
        w.set_paths(Path("/in.mp4"), Path("/out.mp4"))
        w.fail(CoverageUnder("mark extends left +48px"))
        d = json.loads(p.read_text())
        check("a failure writes the outcome", d["outcome"] == "coverage_under")
        check("a failure clears the output path", d["output"] is None,
              str(d["output"]))
        check("the message survives", "left +48px" in d["error"]["message"])
        check("no traceback for a mapped failure", "traceback" not in d["error"])

        p = Path(tmp) / "internal.json"
        w = ReportWriter(p)
        w.fail(RuntimeError("boom"))
        d = json.loads(p.read_text())
        check("an unmapped failure is internal", d["outcome"] == "internal")
        check("an unmapped failure keeps a traceback", "traceback" in d["error"])

        # flush() is the safety net for a path that returned without saying how. A
        # missing file is indistinguishable from a process that died before starting.
        p = Path(tmp) / "flush.json"
        w = ReportWriter(p)
        w.flush()
        check("flush writes a report for a silent exit", p.is_file())
        check("that report is internal, not ok",
              json.loads(p.read_text())["outcome"] == "internal")

        # Terminating twice must not rewrite: the first answer is the true one.
        p = Path(tmp) / "once.json"
        w = ReportWriter(p)
        w.ok()
        w.fail(RuntimeError("later"))
        w.flush()
        check("a report is written once and not overwritten",
              json.loads(p.read_text())["outcome"] == "ok")

        # An unknown outcome must fail loudly here rather than reach a consumer.
        p = Path(tmp) / "bad.json"
        w = ReportWriter(p)
        try:
            w._finish("not-a-real-outcome", ok=False)
            check("an unknown outcome is rejected", False, "no assertion raised")
        except AssertionError:
            check("an unknown outcome is rejected", True)


def test_serializers() -> None:
    print("\n[serializers]")
    import numpy as np

    from wmrm.region import Box
    from wmrm.report import coverage_to_dict, unblend_to_dict

    class FakeCoverage:
        box = Box(1, 2, 3, 4)
        ring_px = 48
        residual_fraction = 0.0129
        reach = {"left": 48, "right": 0}
        suggested = Box(5, 6, 7, 8)
        signal_gradient = True
        signal_variance = False
        inconclusive = False
        ok = False

    d = coverage_to_dict(FakeCoverage())
    check("signal_gradient serializes as a bool",
          isinstance(d["signal_gradient"], bool), repr(d["signal_gradient"]))
    check("signal_variance serializes as a bool",
          isinstance(d["signal_variance"], bool), repr(d["signal_variance"]))
    check("ring_px serializes as an int", isinstance(d["ring_px"], int))
    check("reach values are ints", all(isinstance(v, int) for v in d["reach"].values()))
    check("suggested becomes an object", d["suggested"] == {"x": 5, "y": 6, "w": 7, "h": 8})
    check("the whole thing is JSON-serialisable", json.dumps(d) is not None)

    class FakeUnblend:
        residual_before = 45.8108
        residual = 44.7084
        alpha_scale = 1.2
        opaque = np.zeros((4, 4), bool)

        @property
        def opaque_fraction(self) -> float:
            return float(self.opaque.mean())

    u = unblend_to_dict(FakeUnblend())
    check("unblend has exactly the four fields that exist",
          set(u) == {"residual_before", "residual", "alpha_scale", "opaque_fraction"},
          str(sorted(u)))
    # `background floor 12.24` is a hand-measured constant in the README, not something
    # the code computes. An earlier draft of this schema invented it as a runtime field.
    check("unblend does not invent background_floor", "background_floor" not in u)
    check("unblend is JSON-serialisable", json.dumps(u) is not None)


def test_verify_to_dict() -> None:
    print("\n[verify]")
    from wmrm.verify import VerifyResult

    r = VerifyResult()
    r.add("resolution", True, "480x640 vs 480x640")
    r.add("duration", False, "15.00s vs 14.00s")
    d = r.to_dict()
    check("checks become named objects, not positional triples",
          isinstance(d["checks"][0], dict))
    check("each check has name/passed/detail",
          {"name", "passed", "detail"} <= set(d["checks"][0]))
    check("passed is a real bool", d["checks"][0]["passed"] is True)
    check("ok reflects the failure", d["ok"] is False)
    check("failed lists the failing check names", d["failed"] == ["duration"])
    check("it is JSON-serialisable", json.dumps(d) is not None)


def main() -> int:
    test_outcome_mapping()
    test_report_written_on_every_path()
    test_serializers()
    test_verify_to_dict()
    print(f"\n{len(PASSED)} passed, {len(FAILED)} failed")
    if FAILED:
        for name in FAILED:
            print(f"  FAILED: {name}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
