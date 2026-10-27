"""
Held-rod contact: simulation._contact_face and _resolve_rigid_contact.

Each case is built against player MID_CENTRE of a mid rod, on both teams and
both sides of the rod. x_pre/y_pre are where the ball was at the start of the
substep, as the tick loop passes them.
"""

import pytest

import config
from simulation import _contact_face, _resolve_rigid_contact
from helpers import (MID_CENTRE, T0_MID, T1_MID, ball_at, half_thickness,
                     half_width, hold)

P = MID_CENTRE


@pytest.fixture(params=[T0_MID, T1_MID], ids=["T0", "T1"])
def rod(request, field):
    return hold(field.rods[request.param])


@pytest.fixture(params=[1, -1], ids=["+x", "-x"])
def side(request):
    return request.param


def assert_ejected(ball, rod, axis, exit_sign):
    """After a bounce the ball sits outside the slab, moving away from the face."""
    py = rod.player_positions[P]
    if axis == 'x':
        assert ball.x == pytest.approx(rod.x + exit_sign * half_thickness(rod))
        assert ball.vx * exit_sign > 0
    else:
        assert ball.y == pytest.approx(py + exit_sign * half_width(rod))
        assert ball.vy * exit_sign >= 0
    pen_x = half_thickness(rod) - abs(ball.x - rod.x)
    pen_y = half_width(rod) - abs(ball.y - py)
    assert min(pen_x, pen_y) <= 1e-9


def test_snap_ball_in_reach_box_but_off_slab_is_left_alone(rod, side):
    # abaf4f4: 2 cm off the face is inside the wide reach box but not touching.
    # Before the fix an approaching ball was teleported onto the face.
    ball = ball_at(rod, P, gap=2.0, side=side, vx=-side * 100.0)
    assert rod.player_in_box(ball.x, ball.y, config.BALL_RADIUS) == P
    before = (ball.x, ball.y, ball.vx, ball.vy)
    x_pre = ball.x + side * 1.0

    assert _contact_face(ball, rod, P, x_pre, ball.y) is None
    assert _resolve_rigid_contact(ball, rod, P, x_pre, ball.y, reflect=True) is False
    assert (ball.x, ball.y, ball.vx, ball.vy) == before


def test_front_face_near_edge_is_front_not_side(rod, side):
    # 9ee58fa: a fast ball entering the front near a player's edge ends deeper
    # in x (1.5) than in y (0.2). Least penetration on the current position
    # called that a side hit and let the ball slide on past the man.
    dy = half_width(rod) - 0.2
    ball = ball_at(rod, P, gap=-1.5, side=side, vx=-side * 200.0, dy=dy)
    x_pre = rod.x + side * (half_thickness(rod) + 0.5)

    assert _contact_face(ball, rod, P, x_pre, ball.y) == ('x', side)
    assert _resolve_rigid_contact(ball, rod, P, x_pre, ball.y, reflect=True)
    assert_ejected(ball, rod, 'x', side)
    assert ball.vx == pytest.approx(side * 200.0 * config.CONTROLLED_SLOWDOWN)


@pytest.mark.parametrize("depth", [0.5, 3.0], ids=["past-centre", "out-the-back"])
def test_tunnelled_ball_ejected_on_approach_side(rod, side, depth):
    # Started outside the slab, now past the rod's centre line (or right
    # through the slab but still in the reach box): back out the way it came.
    ball = ball_at(rod, P, gap=0.0, side=side, vx=-side * 250.0)
    ball.x = rod.x - side * depth
    assert rod.player_in_box(ball.x, ball.y, config.BALL_RADIUS) == P
    x_pre = rod.x + side * (half_thickness(rod) + 1.0)

    assert _contact_face(ball, rod, P, x_pre, ball.y) == ('x', side)
    assert _resolve_rigid_contact(ball, rod, P, x_pre, ball.y, reflect=True)
    assert_ejected(ball, rod, 'x', side)


@pytest.mark.parametrize("ysign", [1, -1], ids=["+y", "-y"])
def test_side_approach_is_side_face(rod, ysign):
    py = rod.player_positions[P]
    hw = half_width(rod)
    ball = ball_at(rod, P, gap=-half_thickness(rod), side=1,   # level with the rod centre
                   vx=0.0, vy=-ysign * 40.0, dy=ysign * (hw - 0.5))
    y_pre = py + ysign * (hw + 0.5)

    assert _contact_face(ball, rod, P, ball.x, y_pre) == ('y', ysign)
    assert _resolve_rigid_contact(ball, rod, P, ball.x, y_pre, reflect=True)
    assert_ejected(ball, rod, 'y', ysign)
    assert ball.vy == pytest.approx(ysign * 40.0 * config.CONTROLLED_SLOWDOWN)


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
    "Known bug: the side-face push has no rail check. With the rod slid all the "
    "way over, the gap between the outer man and the rail (2.5 cm) is narrower "
    "than the ball (3.5 cm), so the push puts the ball centre inside the wall."))
@pytest.mark.parametrize("ysign", [1, -1], ids=["+y", "-y"])
def test_side_face_push_stays_off_rail(field, rod, ysign):
    # Slide the rod so its outermost man on the ysign side is as close to that
    # rail as it can get, then bring the ball in from the rail side.
    lo, hi = rod.slide_range
    rod.set_offset(hi if ysign > 0 else lo)
    p = rod.n_players - 1 if ysign > 0 else 0
    py = rod.player_positions[p]
    hw = half_width(rod)
    ball = ball_at(rod, p, gap=-half_thickness(rod), side=1, vy=-ysign * 40.0,
                   dy=ysign * (hw - 0.5))
    y_pre = py + ysign * (hw + 0.2)

    assert _resolve_rigid_contact(ball, rod, p, ball.x, y_pre, reflect=True)
    r = config.BALL_RADIUS
    assert r <= ball.y <= field.width - r


@pytest.mark.parametrize("ysign", [1, -1], ids=["+y", "-y"])
def test_rod_slid_onto_ball_uses_rod_pre_position(rod, ysign):
    # 925bf69: a rod slides 4.5 cm sideways onto a still ball level with its
    # slab: the ball started 3.5 cm off the man's centre (just clear) and now
    # sits 1 cm past it. Judged against the rod's current position the ball
    # looks like it is on the far side, and gets shoved out through the man;
    # the rod's position before the move shows its leading side face hit the
    # ball, which goes out ahead of the man.
    rod.mark_pre()
    rod.set_offset(ysign * 4.5)
    py = rod.player_positions[P]
    ball = ball_at(rod, P, gap=-half_thickness(rod), side=1, dy=-ysign * 1.0)
    assert abs(ball.y - (py - ysign * 4.5)) > half_width(rod)   # clear before the move
    x0 = ball.x

    assert _contact_face(ball, rod, P, ball.x, ball.y) == ('y', ysign)
    assert _resolve_rigid_contact(ball, rod, P, ball.x, ball.y, reflect=True)
    assert ball.x == x0                             # not shoved along x
    assert ball.y == pytest.approx(py + ysign * half_width(rod))
