"""
Own-goal credit (PointResult.is_self_goal).

Rule: only a HIT is a touch. A whiff passes through and changes nothing; a
held-rod PASSIVE block clears the credit (so does the ball stopping, but no
scene can score after a stop without a new touch that sets or clears the
credit itself, so that branch is not observable from a point's result).
"""

import pytest

from helpers import (T0_GOA, T0_MID, T1_GOA, T1_MID, ScriptedStrategy, log_of,
                     run_scene, swing, up_once_moving)


def t0_kick(vx, hands=(), swings=None):
    """T0 shoots from its mid rod (then lifts it); extra held rods sit down."""
    return ScriptedStrategy(hands={T0_MID, *hands}, pos={T0_MID: up_once_moving},
                            swings={T0_MID: swing(vx, 0.0), **(swings or {})})


IDLE_T1 = dict(hands={T1_GOA})   # T1 holds a rod the ball never reaches


def test_hit_into_own_net_is_own_goal(fps):
    _, result = run_scene(t0_kick(-150.0), ScriptedStrategy(**IDLE_T1))
    assert result.winner == 1
    assert result.is_self_goal


def test_passive_block_clears_credit(fps):
    # T0 shoots at T1's held mid, the block sends it back into T0's net
    _, result = run_scene(t0_kick(150.0), ScriptedStrategy(hands={T1_MID}))
    assert log_of(result, 'PASSIVE', 'T1-mid')
    assert result.winner == 1
    assert not result.is_self_goal


def test_whiff_keeps_credit(fps):
    # T0's own goalie swings far too late: the ball slips under it into the net
    t0 = t0_kick(-150.0, hands={T0_GOA}, swings={T0_GOA: swing(150.0, 0.0, 2.0, 2.1)})
    _, result = run_scene(t0, ScriptedStrategy(**IDLE_T1))
    assert not log_of(result, 'PASSIVE', 'T0-goa')
    assert not log_of(result, 'HIT', 'T0-goa')
    assert result.winner == 1
    assert result.is_self_goal


def deflect_scene():
    """T0's shot reflects off T1's FREE (unheld, lowered) mid into T0's net."""
    t1 = ScriptedStrategy(hands={T1_GOA}, passive={T1_MID: (0.0, False)})
    return run_scene(t0_kick(150.0), t1)


def test_deflect_scene_is_valid(fps):
    _, result = deflect_scene()
    assert log_of(result, 'DEFLECT')
    assert len(log_of(result, 'HIT')) == 1          # only the kickoff shot
    assert not log_of(result, 'PASSIVE')
    assert result.winner == 1


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
    "Undecided: a free-rod DEFLECT does not clear last_hit_team, so a shot "
    "that bounces off an opponent's unheld man into the shooter's own net is "
    "an own goal. When this is decided, replace the xfail with the rule."))
def test_free_rod_deflect_clears_credit(fps):
    _, result = deflect_scene()
    assert not result.is_self_goal
