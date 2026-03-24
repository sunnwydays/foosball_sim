"""
strategy.py — Strategy interface and implementations for continuous simulation.

A Strategy is called every tick and makes three decisions:

  choose_hands(team_rods, ball, field, n_hands)
      → set of rod indices to control (up to n_hands).
      Switching to a new rod incurs SWITCH_DELAY on that rod.

  choose_targets(controlled_rods, ball, field)
      → dict of {rod_idx: (target_y, target_x)} for each controlled rod.
      Rods move toward target_y at MOVEMENT_SPEED. target_x sets rotation.

  choose_hit(rod, player_idx, ball, field)
      → None (don't hit) or (hit_vx, hit_vy) velocity to ADD to the ball.
      Called only when a player's bounding box overlaps the ball.
"""

from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

import config
from field import BallState, Field, Rod


class Strategy(ABC):

    @abstractmethod
    def choose_hands(
        self,
        team_rods: list[tuple[int, Rod]],
        ball: BallState,
        field: Field,
        n_hands: int,
    ) -> set[int]:
        """
        Pick which rods (up to n_hands) this team controls this tick.

        Returns a set of global rod indices.
        """

    @abstractmethod
    def choose_targets(
        self,
        controlled_rods: list[tuple[int, Rod]],
        ball: BallState,
        field: Field,
    ) -> dict[int, tuple[float, float]]:
        """
        For each controlled rod, decide where to move.

        Returns {rod_idx: (target_y_offset, target_x_offset)}.
        """

    @abstractmethod
    def choose_hit(
        self,
        rod: Rod,
        player_idx: int,
        ball: BallState,
        field: Field,
    ) -> Optional[tuple[float, float]]:
        """
        Decide whether to hit the ball when a player overlaps it.

        Parameters
        ----------
        rod         : the rod with the overlapping player.
        player_idx  : which player on the rod overlaps.
        ball        : current ball state (position + velocity).
        field       : full field (read-only).

        Returns
        -------
        None                — don't hit (let ball pass / tilt feet up).
        (hit_vx, hit_vy)    — velocity to ADD to the ball.
        """

    def choose_up(
        self,
        rod: Rod,
        ball: BallState,
        field: Field,
    ) -> bool:
        """Decide whether to flip the rod up (ball phases through). Default: never."""
        return False

    def _apply_hit(
        self,
        rod: Rod,
        aim_x: float,
        aim_y: float,
        player_y: float,
        intended_speed: float,
    ) -> Optional[tuple[float, float]]:
        """
        Compute hit velocity toward (aim_x, aim_y) from the player's position,
        with speed and angle noise from rod.speed_std / rod.angle_std.
        Returns None if the aim point is too close to the player.
        """
        dx = aim_x - rod.x
        dy = aim_y - player_y
        if math.sqrt(dx * dx + dy * dy) < 1e-6:
            return None

        actual_speed = float(np.random.normal(intended_speed, rod.speed_std))
        actual_speed = max(10.0, min(config.BALL_MAX_SPEED, actual_speed))

        intended_angle = math.atan2(dy, dx)
        actual_angle = float(np.random.normal(intended_angle, rod.angle_std))

        return (
            actual_speed * math.cos(actual_angle),
            actual_speed * math.sin(actual_angle),
        )

# ---------------------------------------------------------------------------
# SmackBall — simple baseline
# ---------------------------------------------------------------------------

class SmackBall(Strategy):
    """
    Simple reactive strategy:
    - Hands: hold the rods closest to the ball in x.
    - Positioning: slide each rod's center toward ball_y.
    - Hitting: always smack the ball toward the opponent's goal at max speed.

    The hit adds velocity toward the goal center with noise from accuracy/power_consistency.
    """

    def choose_hands(
        self,
        team_rods: list[tuple[int, Rod]],
        ball: BallState,
        field: Field,
        n_hands: int,
    ) -> set[int]:
        sorted_rods = sorted(team_rods, key=lambda ir: abs(ir[1].x - ball.x))
        return {idx for idx, _ in sorted_rods[:n_hands]}

    def choose_targets(
        self,
        controlled_rods: list[tuple[int, Rod]],
        ball: BallState,
        field: Field,
    ) -> dict[int, tuple[float, float]]:
        targets = {}
        for rod_idx, rod in controlled_rods:
            # Slide rod center toward ball_y
            target_y = ball.y - field.width / 2
            targets[rod_idx] = (target_y, 0.0)
        return targets

    def choose_hit(
        self,
        rod: Rod,
        player_idx: int,
        ball: BallState,
        field: Field,
    ) -> Optional[tuple[float, float]]:
        goal = field.goal_for_attacker(rod.team)
        aim_x = goal.x
        aim_y = (goal.y_min + goal.y_max) / 2
        player_y = rod.player_positions[player_idx]
        return self._apply_hit(rod, aim_x, aim_y, player_y, config.BALL_MAX_SPEED * 0.8)

# ---------------------------------------------------------------------------
# AimAtGap — scans for defensive gaps before hitting
# ---------------------------------------------------------------------------

class AimAtGap(Strategy):
    """
    Like SmackBall but aims through the largest gap in the nearest opponent rod
    between the ball and the goal.
    """

    def choose_hands(
        self,
        team_rods: list[tuple[int, Rod]],
        ball: BallState,
        field: Field,
        n_hands: int,
    ) -> set[int]:
        sorted_rods = sorted(team_rods, key=lambda ir: abs(ir[1].x - ball.x))
        return {idx for idx, _ in sorted_rods[:n_hands]}

    def choose_targets(
        self,
        controlled_rods: list[tuple[int, Rod]],
        ball: BallState,
        field: Field,
    ) -> dict[int, tuple[float, float]]:
        targets = {}
        for rod_idx, rod in controlled_rods:
            target_y = ball.y - field.width / 2
            targets[rod_idx] = (target_y, 0.0)
        return targets

    def _find_gap_target(
        self, rod: Rod, ball: BallState, field: Field
    ) -> tuple[float, float]:
        """Find the (aim_x, aim_y) through the largest gap in the nearest defender."""
        team = rod.team
        goal = field.goal_for_attacker(team)

        if team == 0:
            opp_rods = [r for r in field.rods
                        if r.team != team and r.x > rod.x]
            opp_rods.sort(key=lambda r: r.x)
        else:
            opp_rods = [r for r in field.rods
                        if r.team != team and r.x < rod.x]
            opp_rods.sort(key=lambda r: -r.x)

        if not opp_rods:
            return goal.x, (goal.y_min + goal.y_max) / 2

        nearest = opp_rods[0]
        positions = sorted(nearest.player_positions)
        reach = nearest.width / 2

        gaps: list[tuple[float, float]] = []
        first_top = positions[0] - reach
        if first_top > 0:
            gaps.append((0.0, first_top))
        for i in range(len(positions) - 1):
            lo = positions[i] + reach
            hi = positions[i + 1] - reach
            if hi > lo:
                gaps.append((lo, hi))
        last_bot = positions[-1] + reach
        if last_bot < field.width:
            gaps.append((last_bot, field.width))

        if gaps:
            best = max(gaps, key=lambda g: g[1] - g[0])
            return nearest.x, (best[0] + best[1]) / 2

        return goal.x, (goal.y_min + goal.y_max) / 2

    def choose_hit(
        self,
        rod: Rod,
        player_idx: int,
        ball: BallState,
        field: Field,
    ) -> Optional[tuple[float, float]]:
        aim_x, aim_y = self._find_gap_target(rod, ball, field)
        player_y = rod.player_positions[player_idx]
        return self._apply_hit(rod, aim_x, aim_y, player_y, config.BALL_MAX_SPEED * 0.8)

# ---------------------------------------------------------------------------
# HardOffense — trying to hit into the goal, remove obstacles
# ---------------------------------------------------------------------------

class HardOffense(Strategy):
    """
    Like SmackBall but aims through the largest gap in the nearest opponent rod
    between the ball and the goal.
    """

    def choose_hands(
        self,
        team_rods: list[tuple[int, Rod]],
        ball: BallState,
        field: Field,
        n_hands: int,
    ) -> set[int]:
        sorted_rods = sorted(team_rods, key=lambda ir: abs(ir[1].x - ball.x))
        return {idx for idx, _ in sorted_rods[:n_hands]}

    def choose_targets(
        self,
        controlled_rods: list[tuple[int, Rod]],
        ball: BallState,
        field: Field,
    ) -> dict[int, tuple[float, float]]:
        targets = {}
        for rod_idx, rod in controlled_rods:
            target_y = ball.y - field.width / 2
            targets[rod_idx] = (target_y, 0.0)
        return targets

    def choose_hit(
        self,
        rod: Rod,
        player_idx: int,
        ball: BallState,
        field: Field,
    ) -> Optional[tuple[float, float]]:
        goal = field.goal_for_attacker(rod.team)
        aim_x = goal.x
        goal_cy = (goal.y_min + goal.y_max) / 2
        goal_half = (goal.y_max - goal.y_min) / 2
        aim_y = float(np.clip(np.random.normal(goal_cy, goal_half / 2), goal.y_min, goal.y_max))
        player_y = rod.player_positions[player_idx]
        return self._apply_hit(rod, aim_x, aim_y, player_y, config.BALL_MAX_SPEED * 0.8)

    def choose_up(self, rod: Rod, ball: BallState, field: Field) -> bool:
        """Flip rod up if ball is moving away from opponent's goal past this rod."""
        # Ball moving toward own goal (retreating) and rod is ahead of ball
        if rod.team == 0:
            return ball.vx < 0 and rod.x > ball.x
        else:
            return ball.vx > 0 and rod.x < ball.x