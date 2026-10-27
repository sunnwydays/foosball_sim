"""Passive contact: physics.step_ball on uncontrolled (free) rods."""

import pytest

import config
import physics
from physics import step_ball
from helpers import (MID_CENTRE, T0_MID, T1_MID, ball_at, half_thickness,
                     half_width, lift_others)

R = config.BALL_RADIUS
P = MID_CENTRE


@pytest.fixture(params=[T0_MID, T1_MID], ids=["T0", "T1"])
def rod_idx(request):
    return request.param


@pytest.fixture
def rod(field, rod_idx):
    """A free, lowered mid rod; every other rod is lifted out of the way."""
    lift_others(field, keep={rod_idx})
    return field.rods[rod_idx]


@pytest.fixture(params=[1, -1], ids=["+x", "-x"])
def side(request):
    return request.param


def test_forward_tilted_player_reflects_not_slides(field, rod):
    # 270d45b: a forward-tilted free rod let the ball glance (vx scaled but not
    # reversed), so the ball stayed pressed into the face, picking up vy each
    # substep until it slid off the edge into the net.
    forward = 1 if rod.team == 0 else -1
    rod.set_x_offset(forward * config.ROD_X_REACH)
    ball = ball_at(rod, P, gap=0.3, side=forward, vx=-forward * 60.0)
    dt = 1.0 / 30

    contact_ticks = []
    for tick in range(10):
        step_ball(ball, field, dt, ball_radius=R)
        if ball.contacted_player:
            contact_ticks.append(tick)
    assert contact_ticks == [0]             # one touch, then it is gone
    assert ball.vx * forward > 0            # reversed, back the way it came
    assert not rod.up


def test_lifted_rod_is_skipped(field, rod, side):
    rod.up = True
    ball = ball_at(rod, P, gap=0.3, side=side, vx=-side * 60.0)
    for _ in range(5):
        step_ball(ball, field, 1.0 / 30, ball_radius=R)
        assert not ball.contacted_player
    assert (ball.x - rod.x) * side < 0      # passed straight through
    assert ball.vx * side < 0


def test_very_fast_ball_flips_rod_up(field, rod, side):
    speed = config.PASSTHROUGH_SPEED + 10.0
    ball = ball_at(rod, P, gap=0.3, side=side, vx=-side * speed)
    step_ball(ball, field, 1.0 / 30, ball_radius=R)
    assert rod.up
    assert ball.vx * side < 0                # carries on, slowed, not reflected


def test_front_face_near_edge_is_front_not_side(field, rod, side):
    # 410791c: a shot entering the front of a free man near his edge ends
    # deeper in x (0.65) than in y (0.2). Least penetration on the current
    # position read it as a side clip: vy flipped, vx kept, ball into the net.
    # dt keeps the move to one 0.7 cm inner step.
    dy = half_width(rod) - 0.2
    ball = ball_at(rod, P, gap=0.05, side=side, vx=-side * 140.0, dy=dy)
    y0, face_x = ball.y, rod.x + side * half_thickness(rod)   # before pushback moves the rod
    step_ball(ball, field, 0.7 / 140.0, ball_radius=R)

    assert ball.contacted_player
    assert ball.vx * side > 0                                  # front-face reflect
    assert ball.x == pytest.approx(face_x)
    assert ball.y != pytest.approx(rod.player_positions[P] + half_width(rod))
    assert abs(ball.y - y0) < 1.0                              # not pushed off the side


def test_tunnelled_ball_exits_approach_side(field, rod, side):
    # ae9c5ae: the face came from the pre-step position but the push-out used
    # the current dx, so a ball past the man's centre left out the far side.
    # step_ball's 0.75 cm substeps can't cross the centre line, so drive the
    # inner step directly with one 3.5 cm move.
    ball = ball_at(rod, P, gap=0.5, side=side, vx=-side * 140.0)
    face_x = rod.x + side * half_thickness(rod)   # before pushback moves the rod
    physics._step_ball_inner(ball, field, 3.5 / 140.0, config.FRICTION, R,
                             physics.DEFAULT_CONTACT)
    assert ball.contacted_player
    assert ball.x == pytest.approx(face_x)
    assert ball.vx * side > 0


@pytest.mark.parametrize("ysign", [1, -1], ids=["+y", "-y"])
def test_side_face_keeps_vy_pointing_away(field, rod, ysign):
    # ae9c5ae: the side face negated vy, which turned a ball already moving away
    # from the man (here: a rod dropped onto it) back into him.
    ball = ball_at(rod, P, gap=-half_thickness(rod), side=1,
                   vy=ysign * 5.0, dy=ysign * (half_width(rod) - 0.5))
    step_ball(ball, field, 0.01, friction=0.0, ball_radius=R)
    assert ball.contacted_player
    assert ball.vy == pytest.approx(ysign * 5.0)
    assert ball.y == pytest.approx(rod.player_positions[P] + ysign * half_width(rod))
