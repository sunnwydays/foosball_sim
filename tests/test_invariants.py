"""
Seed sweep (slow): run with `pytest -m slow`.

Plays N_SEEDS points for each matchup at fixed, explicit settings (FPS, skills,
kickoff alternating by seed) and checks invariants that hold for every point,
plus the find_seeds detectors against baselines.

Baselines: re-record them only when a change is MEANT to shift the counts, and
put the commit and date in the comment. Cross-check the method against
`python debug_scripts/find_seeds.py 300` (which uses play.py's settings, so its
totals differ from these).
"""

import pytest

import config
from debug_scripts.detectors import back_bounces, face_slides, snaps, track_rigid_contacts
from field import Field
from simulation import simulate_point
from strategy import DefensiveWall, HardOffense, SmackBall, TiltAndGap

pytestmark = pytest.mark.slow

FPS = 30
N_SEEDS = 200
N_DETERMINISM = 20                  # seeds replayed twice per matchup
SKILLS = {0: (0.7, 0.7), 1: (0.6, 0.7)}   # (accuracy, power_consistency), as play.py
MATCHUPS = {
    "HardOffense-TiltAndGap": (HardOffense, TiltAndGap),
    "SmackBall-DefensiveWall": (SmackBall, DefensiveWall),
}
HIT_TOLERANCE = 0.10                # HIT count may drift this far either way

# Recorded on main 1a2b6a6 (passive-face + whiff rework merged), 2026-10-26,
# FPS=30, seeds 0..199, settings above. SmackBall and DefensiveWall never lift
# a held rod for a teammate's shot, hence their high BACK-BOUNCE.
BASELINES = {
    "HardOffense-TiltAndGap":  {"BACK-BOUNCE": 17,   "FACE-SLIDE": 0, "HIT": 1741},
    "SmackBall-DefensiveWall": {"BACK-BOUNCE": 1540, "FACE-SLIDE": 0, "HIT": 1959},
}


def play(matchup, seed):
    s0, s1 = MATCHUPS[matchup]
    field = Field()
    for rod in field.rods:
        rod.accuracy, rod.power_consistency = SKILLS[rod.team]
    with track_rigid_contacts() as calls:
        result = simulate_point(field, {0: s0(), 1: s1()}, kickoff_team=seed % 2,
                                record=True, seed=seed, collect_action_log=True)
    return field, result, calls


@pytest.fixture(scope="module", params=list(MATCHUPS))
def sweep(request):
    """Every point of one matchup: [(seed, field, result, rigid-contact calls)]."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(config, "FPS", FPS)
        mp.setattr(config, "DT", 1.0 / FPS)
        mp.setattr(config, "MAX_TICKS", int(config.MAX_GAME_TIME * FPS))
        points = [(seed, *play(request.param, seed)) for seed in range(N_SEEDS)]
        yield request.param, points


@pytest.fixture
def fps30(monkeypatch):
    """The detectors read config.DT, so keep it at the sweep's FPS while they run."""
    monkeypatch.setattr(config, "FPS", FPS)
    monkeypatch.setattr(config, "DT", 1.0 / FPS)
    monkeypatch.setattr(config, "MAX_TICKS", int(config.MAX_GAME_TIME * FPS))


def test_deterministic(sweep, fps30):
    matchup, points = sweep
    for seed, _, first, _ in points[:N_DETERMINISM]:
        _, again, _ = play(matchup, seed)
        assert (again.winner, again.ticks, again.outcome) == (first.winner, first.ticks, first.outcome)
        assert ([(f.ball_x, f.ball_y) for f in again.frames]
                == [(f.ball_x, f.ball_y) for f in first.frames]), f"seed {seed}"


def test_ball_stays_within_ends(sweep):
    _, points = sweep
    r = config.BALL_RADIUS
    for seed, field, result, _ in points:
        for f in result.frames:
            if f.event and f.event.startswith('goal'):
                continue    # the scoring substep may end past the goal line
            assert -config.GOAL_DEPTH + r - 1e-9 <= f.ball_x <= field.depth + config.GOAL_DEPTH - r + 1e-9, \
                f"seed {seed} t={f.time:.2f}"


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
    "Known bug: a held man's side face pushes the ball to py +- (PLAYER_WIDTH/2 "
    "+ BALL_RADIUS) with no rail check. The outermost man can sit 3.75 cm from "
    "the rail, so the ball centre lands at 0.75 cm, inside the wall (min 1.75). "
    "See test_rigid_contact.py::test_side_face_push_stays_off_rail."))
def test_ball_stays_within_rails(sweep):
    _, points = sweep
    r = config.BALL_RADIUS
    for seed, field, result, _ in points:
        for f in result.frames:
            assert r - 1e-9 <= f.ball_y <= field.width - r + 1e-9, f"seed {seed} t={f.time:.2f}"


def count(points):
    totals = {"SNAP": 0, "BACK-BOUNCE": 0, "FACE-SLIDE": 0, "HIT": 0}
    for _, _, result, calls in points:
        totals["SNAP"] += len(snaps(result, calls))
        totals["BACK-BOUNCE"] += len(back_bounces(result))
        totals["FACE-SLIDE"] += len(face_slides(result))
        totals["HIT"] += sum(1 for e in result.action_log if e.action == "HIT")
    return totals


def test_detectors_within_baseline(sweep, fps30):
    matchup, points = sweep
    got = count(points)
    base = BASELINES[matchup]
    print(f"\n{matchup}: {got}")
    assert got["SNAP"] == 0
    assert got["BACK-BOUNCE"] <= base["BACK-BOUNCE"]
    assert got["FACE-SLIDE"] <= base["FACE-SLIDE"]
    lo, hi = base["HIT"] * (1 - HIT_TOLERANCE), base["HIT"] * (1 + HIT_TOLERANCE)
    assert lo <= got["HIT"] <= hi, f"HITs {got['HIT']} outside [{lo:.0f}, {hi:.0f}]"
