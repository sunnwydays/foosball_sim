"""
simulation.py — Single-point simulation engine.

simulate_point() runs one foosball point from kickoff to either a goal,
a dead ball, or the turn limit. It is the atomic unit that monte_carlo.py
calls thousands of times.

Turn structure
--------------
Pre-point (before the loop, no hands limit):
  All rods call choose_initial_position() to set starting positions.

Each turn:
1. Identify the possessing rod (attacker) and the other team (defender).
2. SIMULTANEOUS decisions — all happen before any randomness:
     a. Attacker: choose_shot()  → (target_angle, target_speed)
     b. Each team: choose_rods_to_move() → up to N_HANDS rods to move
        then choose_position() for each selected rod.
3. Sample actual shot from normal distributions:
     actual_angle = N(target_angle, rod.angle_std)   [skill controls tightness]
     actual_speed = N(target_speed, rod.speed_std)   [consistency controls tightness]
4. Trace the ball via physics.trace_ball().
5. Interpret result:
     'goal'       → point ends, return winner
     'possession' → ball.y set to actual contact y (no snapping), next turn
     'dead'       → point ends, no winner
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
    n_hands:          int = config.N_HANDS,
) -> PointResult:
    """
    Simulate one foosball point.

    Parameters
    ----------
    field            : Field with rods at whatever offsets they currently have.
                       Rod offsets are mutated in-place each turn.
    strategies       : {team_id: Strategy} — one strategy per team.
    starting_rod_idx : which rod begins with possession (kickoff rod).
    n_hands          : max rods each team may reposition per turn.

    Returns
    -------
    PointResult
    """
    # Ball starts at the kickoff rod's x-position, vertically centered
    kickoff_rod = field.rods[starting_rod_idx]
    ball = BallState(
        x                  = kickoff_rod.x,
        y                  = field.depth / 2,
        possessing_rod_idx = starting_rod_idx,
    )

    trajectory: list[tuple[float, float]] = [(ball.x, ball.y)]

    # ------------------------------------------------------------------
    # Pre-point: all rods choose starting positions (no hands limit)
    # ------------------------------------------------------------------
    for rod in field.rods:
        y_off, x_off = strategies[rod.team].choose_initial_position(
            rod, ball.x, ball.y, field
        )
        rod.set_offset(y_off)
        rod.set_x_offset(x_off)

    for turn in range(config.MAX_TURNS):

        poss_rod      = field.rods[ball.possessing_rod_idx]
        attacker_team = poss_rod.team
        defender_team = 1 - attacker_team

        # ------------------------------------------------------------------
        # Step 1 — Simultaneous decisions
        # ------------------------------------------------------------------

        # Attacker picks a shot
        target_angle, target_speed = strategies[attacker_team].choose_shot(
            poss_rod, ball.y, field
        )

        # Each team repositions up to n_hands rods
        for team in (attacker_team, defender_team):
            team_strategy  = strategies[team]
            team_rod_pairs = field.rods_for_team(team)   # [(global_idx, rod), ...]

            rods_to_move = team_strategy.choose_rods_to_move(
                team_rod_pairs, ball.x, ball.y, field, n_hands
            )

            for rod_idx in rods_to_move:
                rod = field.rods[rod_idx]
                desired_y, desired_x = team_strategy.choose_position(
                    rod, ball.x, ball.y, field
                )
                rod.set_offset(desired_y)
                rod.set_x_offset(desired_x)

        # ------------------------------------------------------------------
        # Step 2 — Sample actual shot from distributions
        # ------------------------------------------------------------------
        actual_angle = float(np.random.normal(target_angle, poss_rod.angle_std))
        actual_speed = float(np.random.normal(target_speed, poss_rod.speed_std))
        actual_speed = max(config.MIN_SPEED, actual_speed)

        # ------------------------------------------------------------------
        # Step 3 — Trace ball trajectory
        # ------------------------------------------------------------------
        result: TraceResult = trace_ball(
            ball.x, ball.y, actual_angle, field,
            possessing_rod_idx=ball.possessing_rod_idx,
        )

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
            # Use the actual contact position — no snapping to player center
            ball = BallState(
                x                  = result.x,
                y                  = result.y,
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
