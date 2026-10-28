"""
Held-rod scenes through the real simulate_point loop, at 10 and 30 FPS.

T0 kicks off from its mid rod with an exact scripted shot along y = 34, the
line of the centre man on T1's mid rod (67.3 cm, 14.6 cm downfield). Every rod
the scene does not name is lifted.
"""

import math

import pytest

import config
from field import BallState
from strategy import SmackBall
from helpers import (MID_CENTRE, T0_MID, T1_GOA, T1_MID, ScriptedStrategy,
                     half_thickness, log_of, run_scene, swing, up_once_moving)


def kick(vx, vy=0.0, **t0_kw):
    """T0 strategy: shoot from the kickoff rod, then lift it out of the way."""
    return ScriptedStrategy(hands={T0_MID}, pos={T0_MID: up_once_moving},
                            swings={T0_MID: swing(vx, vy)}, **t0_kw)


def frame_at(result, t):
    """The recorded frame of the tick that starts at game time t."""
    return result.frames[round(t / config.DT)]


def test_fast_shot_cannot_tunnel_held_rod(fps):
    # d1c42a3: held-rod contact was checked once per tick at the ball's end
    # position. At 10 FPS a 250 cm/s shot moves 25 cm a tick, past the 13 cm
    # reach box, and sailed through a well-placed held man.
    field, result = run_scene(kick(250.0), ScriptedStrategy(hands={T1_MID}))
    rod = field.rods[T1_MID]
    face = rod._base_x - half_thickness(rod)

    blocks = log_of(result, 'PASSIVE', 'T1-mid')
    assert blocks, "the held T1 mid never blocked the shot"
    assert not log_of(result, 'HIT', 'T1-mid')
    assert max(f.ball_x for f in result.frames) <= face + 1e-6
    assert result.winner == 1          # bounced back into T0's empty net


@pytest.mark.parametrize("vel", [(-120.0, 30.0), (-300.0, 40.0)], ids=["exact", "clamped"])
def test_swing_in_window_connects(fps, vel):
    t1 = ScriptedStrategy(hands={T1_MID}, swings={T1_MID: swing(*vel, 0.0, 1.0)})
    _, result = run_scene(kick(150.0), t1)

    hits = log_of(result, 'HIT', 'T1-mid')
    assert len(hits) == 1
    hit = hits[0]
    assert 0.0 <= hit.game_time <= 1.0
    assert hit.intended_vel == vel
    speed = math.hypot(*vel)
    scale = min(1.0, config.BALL_MAX_SPEED / speed)
    assert hit.actual_vel == pytest.approx((vel[0] * scale, vel[1] * scale))
    assert not [e for e in log_of(result, 'PASSIVE', 'T1-mid') if e.game_time <= hit.game_time]
    assert frame_at(result, hit.game_time).ball_vx < 0


def test_backswing_lets_ball_pass_then_logs_whiff(fps):
    # The window opens at 0.35 s; the 100 cm/s shot crosses the reach box at
    # about 0.18-0.31 s, under the lifted man. It must not bounce, and the
    # swing expires (at 0.5 s) as a WHIFF before the ball reaches the net.
    t1 = ScriptedStrategy(hands={T1_MID}, swings={T1_MID: swing(-150.0, 0.0, 0.35, 0.40)})
    field, result = run_scene(kick(100.0), t1)
    rod = field.rods[T1_MID]

    assert not log_of(result, 'PASSIVE', 'T1-mid')
    assert not log_of(result, 'HIT', 'T1-mid')
    assert all(f.ball_vx >= 0 for f in result.frames if f.time < 0.5)
    assert any(f.ball_x > rod._base_x + half_thickness(rod) for f in result.frames)

    whiffs = log_of(result, 'WHIFF', 'T1-mid')
    assert len(whiffs) == 1
    assert whiffs[0].game_time > 0.5                      # logged when the swing expires
    assert abs(whiffs[0].ball_pos[0] - rod._base_x) < 4   # where the ball slipped under
    assert result.winner == 0


def test_slow_ball_still_in_reach_is_struck_when_window_opens(fps):
    # A 20 cm/s ball rolls under the man during the backswing and is still in
    # reach at 1.0 s, when the window opens: it is struck, not whiffed.
    t1 = ScriptedStrategy(hands={T1_MID}, swings={T1_MID: swing(-150.0, 0.0, 1.0, 1.1)})
    field, result = run_scene(kick(20.0), t1)
    rod = field.rods[T1_MID]

    # the scene really does touch the man before the window
    in_slab = [f for f in result.frames
               if f.time < 1.0 - config.DT and abs(f.ball_x - rod._base_x) < half_thickness(rod)]
    assert in_slab

    hits = log_of(result, 'HIT', 'T1-mid')
    assert len(hits) == 1
    assert hits[0].game_time >= 1.0 - config.DT - 1e-9    # the tick in which 1.0 s falls
    assert hits[0].actual_vel == (-150.0, 0.0)
    assert not log_of(result, 'WHIFF', 'T1-mid')
    assert not log_of(result, 'PASSIVE', 'T1-mid')


def test_after_swing_is_over_held_rod_blocks(fps):
    # The swing's follow-through ends at 0.11 s, before the ball arrives
    # (~0.2 s): the man is back down and blocks like an idle held rod.
    t1 = ScriptedStrategy(hands={T1_MID}, swings={T1_MID: swing(-150.0, 0.0, 0.0, 0.01)})
    _, result = run_scene(kick(100.0), t1)

    blocks = log_of(result, 'PASSIVE', 'T1-mid')
    assert blocks
    assert not log_of(result, 'HIT', 'T1-mid')
    assert not log_of(result, 'WHIFF', 'T1-mid')
    assert frame_at(result, blocks[0].game_time).ball_vx < 0


def test_rod_in_switch_delay_deflects_passively(fps):
    # d8520af: a newly grabbed rod stays uncontrolled for SWITCH_DELAY, so a
    # ball arriving then is a free-rod DEFLECT (step_ball), not a held block.
    # T1 holds its goalie until the shot is struck, then grabs the mid rod.
    t1 = ScriptedStrategy(hands=lambda ball: {T1_MID} if ball.vx > 0 else {T1_GOA},
                          passive={T1_MID: (0.0, False)})
    field, result = run_scene(kick(100.0), t1)
    rod = field.rods[T1_MID]
    acquired = config.DT                     # tick 1: first tick the ball is moving
    delay_over = acquired + config.SWITCH_DELAY

    deflects = [e for e in log_of(result, 'DEFLECT')
                if abs(e.ball_pos[0] - rod._base_x) < half_thickness(rod) + 1]
    assert deflects
    first = deflects[0]
    assert first.game_time < delay_over
    frame = frame_at(result, first.game_time)
    assert frame.rod_switching[T1_MID] and not frame.rod_ctrl[T1_MID]
    assert not [e for e in log_of(result, 'PASSIVE', 'T1-mid') if e.game_time < delay_over]
    assert frame.ball_vx < 0                  # reflected off the free man


def test_immediate_hit_window(field):
    # A stopped ball already level with the rod: no backswing, the window
    # opens now and lasts the active part of a swing.
    rod = field.rods[T0_MID]
    py = rod.player_positions[MID_CENTRE]
    ball = BallState(x=rod.x, y=py)
    goal = field.goal_for_attacker(0)
    t = 2.0
    s = SmackBall()._commit_swing(rod, ball, field, t, lambda: (goal.x, 34.0), config.HIT_SPEED)
    assert s is not None
    assert s.active_start == t
    assert s.window_end == pytest.approx(t + config.SWING_DURATION * (1 - config.BACKSWING_RATIO))

    # Too fast to see and swing at: no free strike
    fast = BallState(x=rod.x, y=py, vx=config.IMMEDIATE_SWING_MAX_SPEED + 5.0)
    assert SmackBall()._commit_swing(rod, fast, field, t, lambda: (goal.x, 34.0), config.HIT_SPEED) is None
