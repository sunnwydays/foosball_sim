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

import math
from typing import Optional

import config
from field import BallState, Field, Goal


def step_ball(
    ball: BallState,
    field: Field,
    dt: float,
    friction: float = config.FRICTION,
    restitution: float = 1.0,
    ball_radius: float = 0.0,
) -> str:
    """
    Advance the ball by one tick.

    1. Move: pos += vel * dt
    2. Wall bounces: reflect off side walls (y), check end walls (x)
    3. Friction: decelerate toward zero

    Parameters
    ----------
    friction    : deceleration in cm/s² (default: config.FRICTION)
    restitution : bounce dampening 0–1 (1 = perfect elastic, default)
    ball_radius : ball radius in cm for collision offset (default: 0 = point)

    Returns
    -------
    'play'    — ball is still in play
    'goal:0'  — ball entered a goal, team 0 scored
    'goal:1'  — ball entered a goal, team 1 scored
    'stopped' — ball speed dropped below STOP_THRESHOLD
    """
    r = ball_radius

    # --- Move ---
    ball.x += ball.vx * dt
    ball.y += ball.vy * dt

    # --- Side wall bounces (y boundaries) ---
    if ball.y <= r:
        ball.y = 2 * r - ball.y         # reflect off bottom
        ball.vy = abs(ball.vy) * restitution
    elif ball.y >= field.width - r:
        ball.y = 2 * (field.width - r) - ball.y
        ball.vy = -abs(ball.vy) * restitution

    # Clamp in case of floating-point overshoot
    ball.y = max(r, min(field.width - r, ball.y))

    # --- End wall / goal check (x boundaries) ---
    if ball.x <= r:
        if field.left_goal.contains(ball.y):
            return f"goal:{field.left_goal.scoring_team}"
        # Bounce off end wall (outside goal)
        ball.x = 2 * r - ball.x
        ball.vx = abs(ball.vx) * restitution

    elif ball.x >= field.depth - r:
        if field.right_goal.contains(ball.y):
            return f"goal:{field.right_goal.scoring_team}"
        ball.x = 2 * (field.depth - r) - ball.x
        ball.vx = -abs(ball.vx) * restitution

    # Clamp x
    ball.x = max(r, min(field.depth - r, ball.x))

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
        player_idx = rod.player_in_box(ball.x, ball.y)
        if player_idx is not None:
            hits.append((rod_idx, player_idx))
    return hits
