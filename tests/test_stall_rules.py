"""Dead-ball and possession-stall rules."""

from field import BallState
from simulation import _teams_with_reach
from helpers import (T0_MID, T1_GOA, ScriptedStrategy, lift_others, run_scene,
                     swing, up_once_moving)


def test_lifted_rod_counts_as_reach(field):
    # dff3dd8: a ball stopped under a lifted rod ended the point as a dead
    # ball, although the team could lower the rod and play it.
    lift_others(field)
    ball = BallState(x=field.rods[T0_MID]._base_x, y=34.0)
    assert _teams_with_reach(ball, field) == {0}


def test_reach_is_x_only(field):
    rod = field.rods[T0_MID]
    reach = rod.rod_x_reach + rod.thickness / 2 + 1.75
    assert _teams_with_reach(BallState(x=rod._base_x + reach - 0.01, y=1.0), field) == {0}
    # 7.3 cm from both mid rods: nobody's rod can get there
    assert _teams_with_reach(BallState(x=60.0, y=34.0), field) == set()


def creep(vx):
    """T0 nudges the ball off its mid rod, then lifts the rod."""
    t0 = ScriptedStrategy(hands={T0_MID}, pos={T0_MID: up_once_moving},
                          swings={T0_MID: swing(vx, 0.0)})
    return run_scene(t0, ScriptedStrategy(hands={T1_GOA}))


def test_ball_stopped_out_of_reach_is_dead(fps):
    # 12.5 cm/s stops at ~60.2 cm, the strip no rod reaches between the mids
    field, result = creep(12.5)
    last = result.frames[-1]
    assert result.outcome == 'dead_ball'
    assert result.winner is None
    assert _teams_with_reach(BallState(x=last.ball_x, y=last.ball_y), field) == set()


def test_ball_stopped_under_lifted_rod_is_not_dead(fps):
    # 10.5 cm/s stops at ~58 cm, under T0's lifted mid. Only T0 can play it,
    # so it is T0's possession (and T0 never plays it: stallout to T1).
    _, result = creep(10.5)
    assert result.outcome == 'stallout'
    assert result.winner == 1
