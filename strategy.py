"""
strategy.py — Strategy interface and implementations for continuous simulation.

A Strategy is called every tick and makes four decisions:

  choose_hands(team_rods, ball, field, n_hands)
      → set of rod indices to control (up to n_hands).
      Switching to a new rod incurs SWITCH_DELAY on that rod.

  choose_targets(controlled_rods, ball, field)
      → dict of {rod_idx: (target_y, target_x, up)} for each controlled rod.
      Rods move toward target_y at MOVEMENT_SPEED. target_x sets rotation.
      up=True flips the rod up so the ball passes through.

  choose_passive(passive_rods, ball, field)
      → dict of {rod_idx: (target_x, up)} for uncontrolled rods.
      Sets rotation and up/down for rods the team isn't holding.

  choose_hit(rod, player_idx, ball, field)
      → None (don't hit) or (hit_vx, hit_vy) velocity to ADD to the ball.
      Called only when a player's bounding box overlaps the ball.
"""

from __future__ import annotations

import math
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
    ) -> dict[int, tuple[float, float, bool]]:
        """
        For each controlled rod, decide where to move and whether to flip up.

        Returns {rod_idx: (target_y_offset, target_x_offset, up)}.
        x_offset is set instantly (wrist rotation). y_offset moves gradually.
        up=True flips the rod up so the ball passes through.
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

    def choose_passive(
        self,
        passive_rods: list[tuple[int, Rod]],
        ball: BallState,
        field: Field,
    ) -> dict[int, tuple[float, bool]]:
        """
        For each uncontrolled rod, decide rotation and up/down.

        Returns {rod_idx: (target_x_offset, up)}.
        Default: neutral position, not flipped.
        """
        return {idx: (0.0, False) for idx, _ in passive_rods}

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
    ) -> dict[int, tuple[float, float, bool]]:
        targets = {}
        for rod_idx, rod in controlled_rods:
            target_y = ball.y - field.width / 2
            targets[rod_idx] = (target_y, 0.0, False)
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
        return self._apply_hit(rod, aim_x, aim_y, player_y, config.HIT_SPEED)

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
    ) -> dict[int, tuple[float, float, bool]]:
        targets = {}
        for rod_idx, rod in controlled_rods:
            target_y = ball.y - field.width / 2
            targets[rod_idx] = (target_y, 0.0, False)
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
        return self._apply_hit(rod, aim_x, aim_y, player_y, config.HIT_SPEED)

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
    ) -> dict[int, tuple[float, float, bool]]:
        targets = {}
        for rod_idx, rod in controlled_rods:
            target_y = ball.y - field.width / 2

            # Flip rod up if ball is moving away from opponent's goal past this rod
            up = False
            if rod.team == 0:
                up = ball.vx < 0 and rod.x > ball.x
            else:
                up = ball.vx > 0 and rod.x < ball.x

            targets[rod_idx] = (target_y, 0.0, up)
        return targets

    def choose_passive(
        self,
        passive_rods: list[tuple[int, Rod]],
        ball: BallState,
        field: Field,
    ) -> dict[int, tuple[float, bool]]:
        """Flip up offensive rods not being held, so they don't block own shots."""
        result = {}
        for rod_idx, rod in passive_rods:
            if self._is_offensive(rod, field):
                result[rod_idx] = (0.0, True)
            else:
                result[rod_idx] = (0.0, False)
        return result

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
        return self._apply_hit(rod, aim_x, aim_y, player_y, config.HIT_SPEED * 1.5)

    @staticmethod
    def _is_offensive(rod: Rod, field: Field) -> bool:
        """True if the rod is on the opponent's half (closer to the goal it attacks)."""
        mid = field.depth / 2
        if rod.team == 0:
            return rod.x > mid
        else:
            return rod.x < mid


# ---------------------------------------------------------------------------
# DefensiveWall — tilt passive defensive rods forward to block
# ---------------------------------------------------------------------------

class DefensiveWall(Strategy):
    """
    Defensive-focused strategy:
    - Controlled rods track ball_y and stay neutral x.
    - Passive defensive rods tilt forward (toward the ball) to block incoming shots.
    - Passive offensive rods flip up to clear the path for teammates.
    - Hits aim at goal center at full power.
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
    ) -> dict[int, tuple[float, float, bool]]:
        targets = {}
        for rod_idx, rod in controlled_rods:
            target_y = ball.y - field.width / 2
            targets[rod_idx] = (target_y, 0.0, False)
        return targets

    def choose_passive(
        self,
        passive_rods: list[tuple[int, Rod]],
        ball: BallState,
        field: Field,
    ) -> dict[int, tuple[float, bool]]:
        """Defensive rods tilt forward; offensive rods flip up."""
        result = {}
        for rod_idx, rod in passive_rods:
            if self._is_offensive(rod, field):
                # Offensive rod: flip up to clear path
                result[rod_idx] = (0.0, True)
            else:
                # Defensive rod: tilt forward (toward opponent) to block
                tilt = config.ROD_X_REACH if rod.team == 0 else -config.ROD_X_REACH
                result[rod_idx] = (tilt, False)
        return result

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
        return self._apply_hit(rod, aim_x, aim_y, player_y, config.HIT_SPEED)

    @staticmethod
    def _is_offensive(rod: Rod, field: Field) -> bool:
        mid = field.depth / 2
        return (rod.x > mid) if rod.team == 0 else (rod.x < mid)


# ---------------------------------------------------------------------------
# TiltAndGap — defensive tilt + offensive gap-finding
# ---------------------------------------------------------------------------

class TiltAndGap(Strategy):
    """
    Combined strategy:
    - Passive defensive rods tilt forward to block.
    - Passive offensive rods flip up.
    - Controlled rods track ball_y.
    - Hits aim through the largest gap in the nearest defender.
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
    ) -> dict[int, tuple[float, float, bool]]:
        targets = {}
        for rod_idx, rod in controlled_rods:
            target_y = ball.y - field.width / 2

            # Flip up controlled rod if ball is retreating past it
            up = False
            if rod.team == 0:
                up = ball.vx < 0 and rod.x > ball.x
            else:
                up = ball.vx > 0 and rod.x < ball.x

            targets[rod_idx] = (target_y, 0.0, up)
        return targets

    def choose_passive(
        self,
        passive_rods: list[tuple[int, Rod]],
        ball: BallState,
        field: Field,
    ) -> dict[int, tuple[float, bool]]:
        result = {}
        for rod_idx, rod in passive_rods:
            mid = field.depth / 2
            is_offensive = (rod.x > mid) if rod.team == 0 else (rod.x < mid)
            if is_offensive:
                result[rod_idx] = (0.0, True)
            else:
                tilt = config.ROD_X_REACH if rod.team == 0 else -config.ROD_X_REACH
                result[rod_idx] = (tilt, False)
        return result

    def choose_hit(
        self,
        rod: Rod,
        player_idx: int,
        ball: BallState,
        field: Field,
    ) -> Optional[tuple[float, float]]:
        aim_x, aim_y = self._find_gap_target(rod, ball, field)
        player_y = rod.player_positions[player_idx]
        return self._apply_hit(rod, aim_x, aim_y, player_y, config.HIT_SPEED)

    @staticmethod
    def _find_gap_target(
        rod: Rod, ball: BallState, field: Field
    ) -> tuple[float, float]:
        """Find aim point through the largest gap in the nearest defender."""
        team = rod.team
        goal = field.goal_for_attacker(team)

        if team == 0:
            opp_rods = [r for r in field.rods
                        if r.team != team and r.x > rod.x and not r.up]
            opp_rods.sort(key=lambda r: r.x)
        else:
            opp_rods = [r for r in field.rods
                        if r.team != team and r.x < rod.x and not r.up]
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


# ---------------------------------------------------------------------------
# ReactiveBlock — tilts passive rods toward the ball dynamically
# ---------------------------------------------------------------------------

class ReactiveBlock(Strategy):
    """
    Dynamically positions passive rods based on ball location:
    - Passive rods between the ball and own goal tilt forward to block.
    - Passive rods behind the ball (further from own goal) flip up.
    - Controlled rods track ball_y and hit at gap.
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
    ) -> dict[int, tuple[float, float, bool]]:
        targets = {}
        for rod_idx, rod in controlled_rods:
            target_y = ball.y - field.width / 2
            targets[rod_idx] = (target_y, 0.0, False)
        return targets

    def choose_passive(
        self,
        passive_rods: list[tuple[int, Rod]],
        ball: BallState,
        field: Field,
    ) -> dict[int, tuple[float, bool]]:
        result = {}
        for rod_idx, rod in passive_rods:
            # Is this rod between the ball and our own goal?
            if rod.team == 0:
                between = rod.x < ball.x   # our goal is at x=0
            else:
                between = rod.x > ball.x   # our goal is at x=depth

            if between:
                # Defensive position: tilt forward to block
                tilt = config.ROD_X_REACH if rod.team == 0 else -config.ROD_X_REACH
                result[rod_idx] = (tilt, False)
            else:
                # Behind the ball: flip up to clear path
                result[rod_idx] = (0.0, True)
        return result

    def choose_hit(
        self,
        rod: Rod,
        player_idx: int,
        ball: BallState,
        field: Field,
    ) -> Optional[tuple[float, float]]:
        aim_x, aim_y = TiltAndGap._find_gap_target(rod, ball, field)
        player_y = rod.player_positions[player_idx]
        return self._apply_hit(rod, aim_x, aim_y, player_y, config.HIT_SPEED)