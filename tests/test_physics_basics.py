"""Sanity checks on the core physics, with numbers worked out by hand."""

import math

import pytest

import config
from field import BallState, Field
from physics import step_ball
from helpers import T0_MID, half_thickness, half_width, hold, lift_others

R = config.BALL_RADIUS


@pytest.fixture
def empty(field):
    """A field with every rod lifted: the ball only meets walls."""
    lift_others(field)
    return field


def step(ball, field, dt=0.1, friction=config.FRICTION):
    return step_ball(ball, field, dt, friction=friction, ball_radius=R)


# ---------------------------------------------------------------------------
# Ball motion
# ---------------------------------------------------------------------------

def test_free_motion_moves_v_dt(empty):
    ball = BallState(x=30.0, y=20.0, vx=5.0, vy=3.0)
    assert step(ball, empty, friction=0.0) == 'play'
    assert ball.x == pytest.approx(30.5)
    assert ball.y == pytest.approx(20.3)
    assert (ball.vx, ball.vy) == (5.0, 3.0)


def test_friction_takes_friction_dt_off_speed(empty):
    ball = BallState(x=30.0, y=20.0, vx=30.0, vy=40.0)   # speed 50
    step(ball, empty, dt=0.1)
    assert ball.speed == pytest.approx(50.0 - config.FRICTION * 0.1)
    assert ball.vx / ball.vy == pytest.approx(30.0 / 40.0)   # direction kept


def test_friction_below_threshold_stops(empty):
    speed = config.STOP_THRESHOLD + 0.5          # loses 1 cm/s this tick
    ball = BallState(x=30.0, y=20.0, vx=speed)
    assert step(ball, empty, dt=0.1) == 'stopped'
    assert (ball.vx, ball.vy) == (0.0, 0.0)


@pytest.mark.parametrize("wall", ["bottom", "top"])
def test_side_wall_reflects(empty, wall):
    # 2 cm of travel, starting 0.5 cm from the wall: ends 1.5 cm back out
    if wall == "bottom":
        ball = BallState(x=30.0, y=R + 0.5, vy=-20.0)
        expect_y, expect_vy = R + 1.5, 20.0
    else:
        ball = BallState(x=30.0, y=empty.width - R - 0.5, vy=20.0)
        expect_y, expect_vy = empty.width - R - 1.5, -20.0
    step(ball, empty, friction=0.0)
    assert ball.y == pytest.approx(expect_y)
    assert ball.vy == expect_vy
    assert getattr(ball, f"bounce_{wall}")
    assert R <= ball.y <= empty.width - R


@pytest.mark.parametrize("end", ["left", "right"])
def test_end_wall_outside_mouth_bounces(empty, end):
    y = 10.0                                        # below the goal mouth
    assert not empty.left_goal.contains(y)
    if end == "left":
        ball = BallState(x=R + 0.5, y=y, vx=-20.0)
        expect_x, expect_vx = R + 1.5, 20.0
    else:
        ball = BallState(x=empty.depth - R - 0.5, y=y, vx=20.0)
        expect_x, expect_vx = empty.depth - R - 1.5, -20.0
    assert step(ball, empty, friction=0.0) == 'play'
    assert ball.x == pytest.approx(expect_x)
    assert ball.vx == expect_vx
    assert getattr(ball, f"bounce_{end}")


@pytest.mark.parametrize("end", ["left", "right"])
def test_end_wall_inside_mouth_scores(empty, end):
    y = empty.width / 2
    if end == "left":
        ball, goal = BallState(x=10.0, y=y, vx=-30.0), empty.left_goal
    else:
        ball, goal = BallState(x=empty.depth - 10.0, y=y, vx=30.0), empty.right_goal
    floor = -config.GOAL_DEPTH + R
    ceil = empty.depth + config.GOAL_DEPTH - R
    for _ in range(100):
        result = step(ball, empty)
        if result != 'play':
            break   # the scoring step may end past the goal line: the point is over
        assert floor <= ball.x <= ceil            # in play: never deeper than GOAL_DEPTH
    assert result == f"goal:{goal.scoring_team}"
    assert result == ("goal:1" if end == "left" else "goal:0")


def test_substepping_matches_one_big_step(empty):
    # 20 cm in one tick: step_ball splits it into 27 substeps of < 0.75 cm
    ball = BallState(x=20.0, y=50.0, vx=200.0, vy=10.0)
    step(ball, empty, dt=0.1, friction=0.0)
    assert ball.x == pytest.approx(40.0)
    assert ball.y == pytest.approx(51.0)

    # With friction the total loss is still FRICTION * dt
    ball = BallState(x=20.0, y=50.0, vx=200.0, vy=0.0)
    step(ball, empty, dt=0.1)
    assert ball.speed == pytest.approx(200.0 - config.FRICTION * 0.1)


# ---------------------------------------------------------------------------
# Rods
# ---------------------------------------------------------------------------

def test_set_offset_clamps(field):
    rod = field.rods[T0_MID]
    lo, hi = rod.slide_range
    rod.set_offset(1000.0)
    assert rod.y_offset == hi
    rod.set_offset(-1000.0)
    assert rod.y_offset == lo
    rod.set_x_offset(100.0)
    assert rod.x_offset == config.ROD_X_REACH
    rod.set_x_offset(-100.0)
    assert rod.x_offset == -config.ROD_X_REACH


def test_step_movement_speed_limited_no_overshoot(field):
    rod = hold(field.rods[T0_MID])
    rod.target_y = 5.0
    dt = 0.01
    max_step = rod.movement_speed * dt
    prev = rod.y_offset
    for _ in range(100):
        rod.step_movement(dt)
        assert rod.y_offset - prev <= max_step + 1e-9
        assert rod.y_offset <= 5.0
        prev = rod.y_offset
    assert rod.y_offset == 5.0
    assert rod.vy == 0.0


def test_uncontrolled_rod_does_not_move(field):
    rod = field.rods[T0_MID]
    rod.target_y = 5.0
    rod.step_movement(0.1)
    assert rod.y_offset == 0.0


def test_rod_in_switch_delay_does_not_move(field):
    rod = hold(field.rods[T0_MID])
    rod.switch_timer = config.SWITCH_DELAY
    rod.target_y = 5.0
    rod.step_movement(0.1)
    assert rod.y_offset == 0.0
    assert rod.switch_timer == pytest.approx(config.SWITCH_DELAY - 0.1)


def test_player_in_box_edges(field):
    rod = field.rods[T0_MID]
    rod.set_x_offset(2.0)   # the box is centred on the base x, not the tilted x
    py = rod.player_positions[2]
    reach_x = rod.rod_x_reach + rod.thickness / 2 + R
    reach_y = rod.width / 2 + R
    eps = 1e-6
    base = rod._base_x
    assert rod.player_in_box(base + reach_x - eps, py, R) == 2
    assert rod.player_in_box(base - reach_x + eps, py, R) == 2
    assert rod.player_in_box(base + reach_x + eps, py, R) is None
    assert rod.player_in_box(base - reach_x - eps, py, R) is None
    assert rod.player_in_box(base, py + reach_y - eps, R) == 2
    assert rod.player_in_box(base, py + reach_y + eps, R) is None
    # reach_y is the slab's half-width: y has no extra reach, only x does
    assert reach_y == half_width(rod)
    assert reach_x > half_thickness(rod)


# ---------------------------------------------------------------------------
# Field layout
# ---------------------------------------------------------------------------

def test_field_layout():
    field = Field()
    assert len(field.rods) == 8
    for team in (0, 1):
        assert len(field.rods_for_team(team)) == 4
    for rod, (team, n) in zip(field.rods, config.ROD_CONFIGS):
        assert (rod.team, rod.n_players) == (team, n)

    nearest_left = min(field.rods, key=lambda r: r.x)
    nearest_right = max(field.rods, key=lambda r: r.x)
    assert (nearest_left.team, nearest_left.n_players) == (0, 3)     # T0 goalie
    assert (nearest_right.team, nearest_right.n_players) == (1, 3)   # T1 goalie

    for goal in (field.left_goal, field.right_goal):
        assert math.isclose(goal.y_min + goal.y_max, field.width)
        assert math.isclose(goal.y_max - goal.y_min, config.GOAL_WIDTH)
    assert field.left_goal.scoring_team == 1
    assert field.right_goal.scoring_team == 0
