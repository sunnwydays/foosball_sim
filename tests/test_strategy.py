"""Strategy unit tests."""

import pytest

import config
from field import BallState
from strategy import HardOffense, SmackBall, TiltAndGap
from helpers import T0_FWD, T1_FWD, half_thickness

FAST = 150.0
SLOW = config.STOP_THRESHOLD - 1.0


# (offset of the ball from the rod along the team's attack direction,
#  ball speed along the attack direction, expected)
CASES = [
    (-10.0, FAST, True),    # behind the rod, coming at it toward the goal
    (2.0, FAST, True),      # inside the slab, not yet past the front face
    (3.0, FAST, False),     # cleared the front face
    (-10.0, -FAST, False),  # behind, moving back toward own goal
    (-10.0, SLOW, False),   # behind, but (nearly) stopped
    (-10.0, 0.0, False),
]


@pytest.mark.parametrize("team,rod_idx", [(0, T0_FWD), (1, T1_FWD)], ids=["T0", "T1"])
@pytest.mark.parametrize("along,v,expected", CASES)
def test_shot_from_behind(field, team, rod_idx, along, v, expected):
    rod = field.rods[rod_idx]
    d = 1 if team == 0 else -1
    ball = BallState(x=rod.x + d * along, y=34.0, vx=d * v)
    assert SmackBall()._shot_from_behind(rod, ball) is expected


def test_shot_from_behind_margin_is_slab(field):
    rod = field.rods[T0_FWD]
    ht = half_thickness(rod)
    s = SmackBall()
    assert s._shot_from_behind(rod, BallState(x=rod.x + ht - 0.01, y=34.0, vx=FAST))
    assert not s._shot_from_behind(rod, BallState(x=rod.x + ht + 0.01, y=34.0, vx=FAST))


@pytest.mark.parametrize("strat", [HardOffense, TiltAndGap])
@pytest.mark.parametrize("team,rod_idx", [(0, T0_FWD), (1, T1_FWD)], ids=["T0", "T1"])
def test_held_rod_lifts_for_teammate_shot(field, strat, team, rod_idx):
    rod = field.rods[rod_idx]
    d = 1 if team == 0 else -1
    coming = BallState(x=rod.x - d * 20.0, y=34.0, vx=d * FAST)
    going_back = BallState(x=rod.x - d * 20.0, y=34.0, vx=-d * FAST)
    assert strat().choose_pos([(rod_idx, rod)], coming, field)[rod_idx][2] is True
    assert strat().choose_pos([(rod_idx, rod)], going_back, field)[rod_idx][2] is False
