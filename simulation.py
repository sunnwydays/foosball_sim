"""
simulation.py — Single-point simulation engine.

simulate_point() runs one foosball point from kickoff to either a goal,
a dead ball, or the turn limit. It is the atomic unit that monte_carlo.py
calls thousands of times.

Turn structure (per turn)
-------------------------
1. Identify the possessing rod and its team (attacker) + the other team (defender).
2. SIMULTANEOUS decisions — both happen before any randomness:
     a. Attacker strategy: choose_shot()     → (target_angle, target_speed)
     b. All rods (both teams): choose_position() → set y_offset
3. Sample the actual shot from normal distributions:
     actual_angle = N(target_angle, rod.angle_std)   [rod skill controls tightness]
     actual_speed = N(target_speed, rod.speed_std)   [rod consistency controls tightness]
4. Trace the ball via physics.trace_ball().
5. Interpret result:
     - 'goal'       → point ends, return winner
     - 'possession' → update ball state, snap to player center, next turn
     - 'dead'       → point ends, no winner

Why simultaneous?
-----------------
In real foosball, you commit to a shot and a defensive position at the same
time — you don't see the opponent's choice before responding. The simultaneous
model captures this: defenders must anticipate, not react.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field as dc_field
from typing import Optional

import numpy as np

import config
from field import Field, BallState, TraceResult
from physics import trace_ball
from strategy import Strategy


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class PointResult:
    """
    The outcome of a single simulated point.

    winner     : 0 or 1 (team that scored), or None (dead ball / turn limit).
    turns      : number of possession changes before the point ended.
    trajectory : list of (x, y) positions — one entry per possession transfer,
                 starting from the kickoff position. Useful for visualisation
                 and debugging (e.g. heatmaps of where interceptions happen).
    """
    winner:     Optional[int]
    turns:      int
    trajectory: list[tuple[float, float]] = dc_field(default_factory=list)


# ---------------------------------------------------------------------------
# Core simulation function
# ---------------------------------------------------------------------------

def simulate_point(
    field:            Field,
    strategies:       dict[int, Strategy],
    starting_rod_idx: int,
) -> PointResult:
    """
    Simulate one foosball point.

    Parameters
    ----------
    field            : Field with rods at whatever y_offsets they currently have.
                       Rod offsets are mutated in-place each turn.
    strategies       : {team_id: Strategy} — one strategy per team.
    starting_rod_idx : which rod begins with possession (kickoff rod).

    Returns
    -------
    PointResult
    """
    # Ball starts at the kickoff rod's x-position, vertically centered
    kickoff_rod = field.rods[starting_rod_idx]
    ball = BallState(
        x                  = kickoff_rod.x,
        y                  = field.height / 2,
        possessing_rod_idx = starting_rod_idx,
    )

    trajectory: list[tuple[float, float]] = [(ball.x, ball.y)]

    for turn in range(config.MAX_TURNS):

        poss_rod      = field.rods[ball.possessing_rod_idx]
        attacker_team = poss_rod.team
        defender_team = 1 - attacker_team

        attacker_strategy = strategies[attacker_team]
        defender_strategy = strategies[defender_team]

        # ------------------------------------------------------------------
        # Step 1 — Simultaneous decisions
        # ------------------------------------------------------------------

        # Attacker picks a shot
        target_angle, target_speed = attacker_strategy.choose_shot(
            poss_rod, ball.y, field
        )

        # ALL rods (both teams) choose their y-positions at the same time.
        # Own-team non-possessing rods position to receive passes;
        # opponent rods position to intercept.
        for rod in field.rods:
            team_strategy = strategies[rod.team]
            desired_y, desired_x = team_strategy.choose_position(
                rod, ball.x, ball.y, field
            )
            rod.set_offset(desired_y)
            rod.set_x_offset(desired_x)

        # ------------------------------------------------------------------
        # Step 2 — Sample actual shot from distributions
        #
        # The normal distribution models human imprecision:
        #   angle_std ∝ 1/skill       (tight = skilled, wide = beginner)
        #   speed_std ∝ 1/consistency (tight = consistent, wide = erratic)
        # ------------------------------------------------------------------
        actual_angle = float(np.random.normal(target_angle, poss_rod.angle_std))
        actual_speed = float(np.random.normal(target_speed, poss_rod.speed_std))
        actual_speed = max(config.MIN_SPEED, actual_speed)   # clamp to valid range

        # ------------------------------------------------------------------
        # Step 3 — Trace ball trajectory
        # ------------------------------------------------------------------
        result: TraceResult = trace_ball(ball.x, ball.y, actual_angle, field)

        # ------------------------------------------------------------------
        # Step 4 — Interpret result
        # ------------------------------------------------------------------
        if result.kind == 'goal':
            return PointResult(
                winner     = result.scoring_team,
                turns      = turn + 1,
                trajectory = trajectory,
            )

        elif result.kind == 'possession':
            # Snap ball to the intercepting player's center for a clean origin
            new_rod    = field.rods[result.rod_idx]
            player_idx = new_rod.player_at_y(result.y)
            if player_idx is not None:
                snapped_y = new_rod.player_positions[player_idx]
            else:
                snapped_y = result.y   # fallback (shouldn't happen)

            ball = BallState(
                x                  = result.x,
                y                  = snapped_y,
                possessing_rod_idx = result.rod_idx,
            )
            trajectory.append((ball.x, ball.y))

        else:   # 'dead'
            return PointResult(
                winner     = None,
                turns      = turn + 1,
                trajectory = trajectory,
            )

    # Reached MAX_TURNS without resolution
    return PointResult(winner=None, turns=config.MAX_TURNS, trajectory=trajectory)
