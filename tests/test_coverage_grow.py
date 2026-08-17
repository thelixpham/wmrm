#!/usr/bin/env python
"""Coverage grow-loop tests. Run directly: `python tests/test_coverage_grow.py`.

Fast and offline -- `check_coverage` is replaced with a script of verdicts, because what
is being pinned here is the loop around it, not the statistics inside it. Those are
measured on real footage and cannot be asserted from a fixture.

The behaviour under test, and why each part of it exists:

- The check only sees `ring` px past the box, so its `reach` saturates there and its
  suggestion comes back a floor rather than a distance. One round is therefore not
  enough on a mark whose faint half detection missed -- measured on MOGI-108, detection
  returned 1654,45,183,63 against a mark starting 101 px further left, `ring=48` reported
  the cap and `ring=183` measured +94.
- Growing applies to detection's guesses only. A box a human typed is a decision.
- A round that quadruples the area is the check latching onto background structure, not
  mark: `reach` is the bounding box of every flagged pixel with no clustering behind it,
  so a single static edge produces it.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wmrm import cli                                        # noqa: E402
from wmrm.coverage import Coverage                          # noqa: E402
from wmrm.region import Box                                 # noqa: E402

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    (PASSED if condition else FAILED).append(name)
    print(f"  {'PASS' if condition else 'FAIL'}  {name}" + (f" -- {detail}" if detail else ""))


def verdict(box: Box, suggested: Box | None, *, inconclusive: bool = False) -> Coverage:
    """A Coverage that is `ok` exactly when nothing is suggested."""
    reach = {"left": 0, "right": 0, "top": 0, "bottom": 0}
    if suggested is not None:
        reach["left"] = box.x - suggested.x
    return Coverage(
        box=box, ring_px=cli._coverage_ring(box), residual_fraction=0.012,
        reach=reach, suggested=suggested, signal_gradient=suggested is not None,
        signal_variance=False, inconclusive=inconclusive,
    )


class Script:
    """Stands in for `check_coverage`, answering from a list and recording the rings."""

    def __init__(self, *steps: Box | None) -> None:
        self.steps = list(steps)
        self.rings: list[int] = []
        self.boxes: list[tuple[int, int, int, int]] = []

    def __call__(self, src, box, *, ring=48, **kw) -> Coverage:
        self.rings.append(ring)
        self.boxes.append(box.as_tuple())
        nxt = self.steps.pop(0) if self.steps else None
        return verdict(box, nxt)


def run(script: Script, box: Box, *, grow: bool):
    """Call the loop with `check_coverage` swapped out for the script."""
    import wmrm.coverage as coverage_mod

    original = coverage_mod.check_coverage
    coverage_mod.check_coverage = script
    try:
        return cli._grow_to_cover(Path("unused.mp4"), box, grow=grow)
    finally:
        coverage_mod.check_coverage = original


def test_ring_scales_with_box() -> None:
    print("\n[ring width]")
    check("a narrow box still gets the 48px floor",
          cli._coverage_ring(Box(0, 0, 20, 20)) == 48)
    check("the ring follows the box width",
          cli._coverage_ring(Box(0, 0, 183, 63)) == 183,
          "MOGI-108: 48 reported the cap, 183 measured +94")
    check("and is capped, because cost scales with the sampled area",
          cli._coverage_ring(Box(0, 0, 900, 63)) == 200)
    check("an explicit --coverage-ring wins over all of that",
          cli._coverage_ring(Box(0, 0, 183, 63), 400) == 400)


def test_saturation_is_reported() -> None:
    """Both cases here are real measurements on MOGI-108, box 1654,45,183,63."""
    print("\n[saturation]")
    from wmrm.report import coverage_to_dict

    floor = verdict(Box(1654, 45, 183, 63), Box(1606, 45, 231, 63))
    floor = Coverage(**{**floor.__dict__, "ring_px": 48,
                        "reach": {"left": 48, "right": 0, "top": 0, "bottom": 0}})
    measured = Coverage(**{**floor.__dict__, "ring_px": 183,
                           "reach": {"left": 94, "right": 0, "top": 0, "bottom": 0},
                           "suggested": Box(1560, 45, 277, 63)})

    check("a reach that fills the ring is a floor, not a distance", floor.saturated,
          "ring 48 -> left +48, and the suggestion was still 52px short")
    check("a reach with room to spare is a measurement", not measured.saturated,
          "ring 183 -> left +94, and the suggestion covered the mark")
    check("the text says so", "at least" in floor.describe())
    check("and does not say so when it measured", "at least" not in measured.describe())
    check("the report carries it, so a service can tell the two apart",
          coverage_to_dict(floor)["saturated"] is True
          and coverage_to_dict(measured)["saturated"] is False)

    edge = Coverage(**{**floor.__dict__, "ring_px": 48,
                       "reach": {"left": 12, "right": 0, "top": 0, "bottom": 0}})
    check("a short reach is never saturated", not edge.saturated)


def test_grows_until_covered() -> None:
    print("\n[growing]")
    # MOGI-108's shape: detection returns the badge alone, one round reaches the mark.
    detected = Box(1654, 45, 183, 63)
    grown = Box(1560, 45, 277, 63)
    script = Script(grown, None)
    box, cov = run(script, detected, grow=True)
    check("it adopts the suggestion", box.as_tuple() == grown.as_tuple(),
          f"got {box.as_tuple()}")
    check("and re-checks it", script.boxes == [detected.as_tuple(), grown.as_tuple()],
          f"checked {script.boxes}")
    check("the verdict returned is the one for the final box", cov.ok)
    check("the ring widens with the box, up to the cap", script.rings == [183, 200],
          f"rings {script.rings}")


def test_multiple_rounds_and_the_cap() -> None:
    print("\n[round limit]")
    # Never converges: every round suggests 20px more. The loop must still stop.
    box = Box(1700, 44, 120, 62)
    script = Script(*[Box(1700 - 20 * i, 44, 120 + 20 * i, 62) for i in range(1, 12)])
    final, cov = run(script, box, grow=True)
    check("it stops after the round limit",
          len(script.boxes) == cli._GROW_ROUNDS, f"{len(script.boxes)} rounds")
    check("and reports the box as still short", not cov.ok)
    # One fewer growth than rounds: the last pass measures, it does not grow, so the
    # box handed back is always one a verdict was actually measured on.
    grown_to = (1700 - 20 * (cli._GROW_ROUNDS - 1), 44,
                120 + 20 * (cli._GROW_ROUNDS - 1), 62)
    check("having grown it as far as it got", final.as_tuple() == grown_to,
          f"got {final.as_tuple()}")
    check("and the verdict belongs to the box returned",
          cov.box.as_tuple() == final.as_tuple(),
          f"verdict on {cov.box.as_tuple()}, returned {final.as_tuple()}")


def test_runaway_growth_is_refused() -> None:
    print("\n[runaway guard]")
    box = Box(1720, 44, 116, 62)
    huge = Box(1626, 0, 294, 222)          # 9.1x -- what input.mp4 actually produces
    script = Script(huge, None)
    final, cov = run(script, box, grow=True)
    check("a box that quadruples is not adopted", final.as_tuple() == box.as_tuple(),
          f"got {final.as_tuple()}")
    check("it stops there rather than checking again", len(script.boxes) == 1)
    check("and the verdict stays UNDER-COVERED", not cov.ok)


def test_grow_off_is_one_measurement() -> None:
    print("\n[grow=False]")
    box = Box(1654, 45, 183, 63)
    script = Script(Box(1560, 45, 277, 63))
    final, cov = run(script, box, grow=False)
    check("a box that was not detected is left alone",
          final.as_tuple() == box.as_tuple(), f"got {final.as_tuple()}")
    check("and measured exactly once", len(script.boxes) == 1)
    check("the suggestion still reaches the caller", cov.suggested is not None)


def test_inconclusive_stops_immediately() -> None:
    print("\n[inconclusive]")
    box = Box(1654, 45, 183, 63)

    def script(src, b, *, ring=48, **kw):
        script.calls = getattr(script, "calls", 0) + 1
        return verdict(b, Box(1560, 45, 277, 63), inconclusive=True)

    import wmrm.coverage as coverage_mod

    original = coverage_mod.check_coverage
    coverage_mod.check_coverage = script
    try:
        final, cov = cli._grow_to_cover(Path("unused.mp4"), box, grow=True)
    finally:
        coverage_mod.check_coverage = original

    check("an inconclusive verdict is not grown from",
          final.as_tuple() == box.as_tuple(), f"got {final.as_tuple()}")
    check("and is not retried", script.calls == 1)
    check("the caller still sees it was inconclusive", cov.inconclusive)


def test_ring_override_reaches_the_check() -> None:
    print("\n[--coverage-ring]")
    script = Script(None)
    run_with = {"grow": True, "ring": 400}
    import wmrm.coverage as coverage_mod

    original = coverage_mod.check_coverage
    coverage_mod.check_coverage = script
    try:
        cli._grow_to_cover(Path("unused.mp4"), Box(1654, 45, 183, 63), **run_with)
    finally:
        coverage_mod.check_coverage = original
    check("the override is what the check is called with", script.rings == [400],
          f"rings {script.rings}")


def main() -> int:
    test_ring_scales_with_box()
    test_saturation_is_reported()
    test_ring_override_reaches_the_check()
    test_grows_until_covered()
    test_multiple_rounds_and_the_cap()
    test_runaway_growth_is_refused()
    test_grow_off_is_one_measurement()
    test_inconclusive_stops_immediately()
    print(f"\n{len(PASSED)} passed, {len(FAILED)} failed")
    for name in FAILED:
        print(f"  FAILED: {name}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
