"""
physics.py — Per-tick ball physics for the time-stepped simulation.

Functions
---------
step_ball(ball, field, dt)
    Move the ball for one tick: apply velocity, friction, wall bounces.
    Returns 'play', 'goal', or 'stopped'.

find_overlapping_players(ball, field)
    Return all (rod_idx, player_idx) pairs whose bounding box overlaps
    the ball's current position.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

import config
from field import BallState, Field


@dataclass
class ContactParams:
    """Tunable parameters for passive player contact."""
    slowdown:          float = config.PASSIVE_SLOWDOWN
    min_speed:         float = config.PASSIVE_MIN_SPEED
    speed_pushback:    float = config.SPEED_PUSHBACK
    offset_pushback:   float = config.OFFSET_PUSHBACK
    pushback_x_start:  float = config.PUSHBACK_X_START
    passthrough_speed: float = config.PASSTHROUGH_SPEED
    rod_up_threshold:  float = config.ROD_UP_THRESHOLD
    glance_threshold:  float = config.GLANCE_THRESHOLD
    deflection:        float = config.PASSIVE_DEFLECTION


# Default instance reused when no overrides are needed
DEFAULT_CONTACT = ContactParams()


def step_ball(
    ball: BallState,
    field: Field,
    dt: float,
    friction: float = config.FRICTION,
    ball_radius: float = 0.0,
    contact: ContactParams = DEFAULT_CONTACT,
    pos_grid: Optional[np.ndarray] = None,
) -> str:
    """
    Advance the ball by one tick.

    1. Move: pos += vel * dt
    2. Wall bounces: reflect off side walls (y), check end walls (x)
    3. Friction: decelerate toward zero

    Parameters
    ----------
    friction    : deceleration in cm/s² (default: config.FRICTION)
    ball_radius : ball radius in cm for collision offset (default: 0 = point)

    Returns
    -------
    'play'    — ball is still in play
    'goal:0'  — ball entered a goal, team 0 scored
    'goal:1'  — ball entered a goal, team 1 scored
    'stopped' — ball speed dropped below STOP_THRESHOLD
    """
    # Reset per-tick contact/bounce flags; inner step(s) set them on real events
    ball.contacted_player = False
    ball.bounce_left = ball.bounce_right = False
    ball.bounce_top = ball.bounce_bottom = False

    # Substep if the ball would move too far in one tick
    max_step = config.PLAYER_THICKNESS / 2
    dist = ball.speed * dt
    if dist > max_step:
        n_sub = int(dist / max_step) + 1
        sub_dt = dt / n_sub
        for _ in range(n_sub):
            result = _step_ball_inner(ball, field, sub_dt, friction,
                                      ball_radius, contact, pos_grid)
            if result != 'play':
                return result
        return 'play'

    return _step_ball_inner(ball, field, dt, friction, ball_radius, contact, pos_grid)


def _step_ball_inner(
    ball: BallState,
    field: Field,
    dt: float,
    friction: float,
    ball_radius: float,
    contact: ContactParams,
    pos_grid: Optional[np.ndarray] = None,
) -> str:
    r = ball_radius

    # --- Move ---
    ball.x += ball.vx * dt
    ball.y += ball.vy * dt

    # --- Accumulate ball position into grid (per substep) ---
    if pos_grid is not None:
        ix = int(ball.x * pos_grid.shape[0] / field.depth)
        iy = int(ball.y * pos_grid.shape[1] / field.width)
        if 0 <= ix < pos_grid.shape[0] and 0 <= iy < pos_grid.shape[1]:
            pos_grid[ix, iy] += dt

    # --- Side wall bounces (y boundaries) ---
    if ball.y <= r:
        ball.y = 2 * r - ball.y         # reflect off bottom
        ball.vy = abs(ball.vy)
        ball.bounce_bottom = True
    elif ball.y >= field.width - r:
        ball.y = 2 * (field.width - r) - ball.y
        ball.vy = -abs(ball.vy)
        ball.bounce_top = True

    # Clamp in case of floating-point overshoot
    ball.y = max(r, min(field.width - r, ball.y))

    # --- End wall / goal check (x boundaries) ---
    gd = config.GOAL_DEPTH

    if ball.x <= r:
        if field.left_goal.contains(ball.y):
            # Ball is in goal opening — score once deep enough
            if ball.x <= -gd + r:
                return f"goal:{field.left_goal.scoring_team}"
            # Otherwise let it keep moving into the goal area
        else:
            # Bounce off end wall (outside goal)
            ball.x = 2 * r - ball.x
            ball.vx = abs(ball.vx)
            ball.bounce_left = True

    elif ball.x >= field.depth - r:
        if field.right_goal.contains(ball.y):
            if ball.x >= field.depth + gd - r:
                return f"goal:{field.right_goal.scoring_team}"
        else:
            ball.x = 2 * (field.depth - r) - ball.x
            ball.vx = -abs(ball.vx)
            ball.bounce_right = True

    # Clamp x (allow ball into goal area but not beyond goal depth)
    ball.x = max(-gd + r, min(field.depth + gd - r, ball.x))

    # --- Passive player contact (uncontrolled rods) ---
    for rod in field.rods:
        if rod.up or rod.controlled:
            continue
        ht = rod.thickness / 2 + r   # half-thickness in x including ball radius
        hw = rod.width / 2 + r       # half-width in y including ball radius
        for py in rod.player_positions:
            dx = ball.x - rod.x
            dy = ball.y - py
            if abs(dx) >= ht or abs(dy) >= hw:
                continue

            # A figurine is overlapping the ball this tick — real contact
            ball.contacted_player = True

            # Which face did the ball enter from?
            pen_x = ht - abs(dx)
            pen_y = hw - abs(dy)

            # --- Side face (y): simple bounce ---
            if pen_x >= pen_y:
                ball.y = py + (hw if dy > 0 else -hw)
                ball.vy = -ball.vy
                continue

            # --- Front face (x): passive contact model ---
            
            # Push ball out of the player
            ball.x = rod.x + (ht if dx > 0 else -ht)

            orig_speed = abs(ball.vx)

            # 1. Rod response
            # Fast ball flips rod up immediately
            if orig_speed > contact.passthrough_speed:
                rod.up = True
                ball.vx *= contact.slowdown
                continue

            hit_sign = 1.0 if dx > 0 else -1.0
            push_dir = -hit_sign
            offset_toward_ball = rod.x_offset * hit_sign  # > 0 is forward tilt

            # Pushback if rod isn't tilted too far forward
            if offset_toward_ball < contact.pushback_x_start:
                offset_dist = contact.pushback_x_start - offset_toward_ball
                pb = orig_speed * contact.speed_pushback + offset_dist * contact.offset_pushback
                rod.set_x_offset(rod.x_offset + pb * push_dir)
                offset_toward_ball = rod.x_offset * hit_sign
            
            # Rod is too tilted for the ball to hit
            if offset_toward_ball < contact.rod_up_threshold:
                rod.up = True

            # 2. Ball response
            reflect_x = offset_toward_ball <= contact.glance_threshold

            if reflect_x:
                ball.vx *= -contact.slowdown
            else:
                ball.vx *= contact.slowdown

            # vy deflection based on where on the face the ball hit
            # dy/hw is -1 to 1: above center → positive, below → negative
            ball.vy += (dy / hw) * orig_speed * contact.deflection

            # Catch slow ball
            if abs(ball.vx) < contact.min_speed:
                ball.vx = 0.0
                ball.vy = 0.0

    # --- Friction ---
    speed = ball.speed
    if speed > 0:
        decel = friction * dt
        if decel >= speed:
            ball.vx = 0.0
            ball.vy = 0.0
        else:
            factor = (speed - decel) / speed
            ball.vx *= factor
            ball.vy *= factor

    # --- Check stopped ---
    if ball.speed < config.STOP_THRESHOLD:
        ball.vx = 0.0
        ball.vy = 0.0
        return 'stopped'

    return 'play'


def find_overlapping_players(
    ball: BallState,
    field: Field,
) -> list[tuple[int, int]]:
    """
    Find all players whose bounding box overlaps the ball's position.

    Returns
    -------
    List of (rod_idx, player_idx) pairs.
    """
    hits: list[tuple[int, int]] = []
    for rod_idx, rod in enumerate(field.rods):
        if rod.up:
            continue
        player_idx = rod.player_in_box(ball.x, ball.y, ball_radius=config.BALL_RADIUS)
        if player_idx is not None:
            hits.append((rod_idx, player_idx))
    return hits
