"""
strategy.py — Strategy interface and implementations for continuous simulation.

A Strategy is called every tick and makes four decisions:

  choose_hands(team_rods, ball, field, n_hands)
      → set of rod indices to control (up to n_hands).
      Switching to a new rod incurs SWITCH_DELAY on that rod.

  choose_pos(controlled_rods, ball, field)
      → dict of {rod_idx: (target_y, target_x, up)} for each controlled rod.
      Rods move toward target_y at MOVEMENT_SPEED. target_x sets rotation.
      up=True flips the rod up so the ball passes through.

  choose_passive(passive_rods, ball, field)
      → dict of {rod_idx: (target_x, up)} for uncontrolled rods.
      Sets rotation and up/down for rods the team isn't holding.

  choose_hit(rod, ball, field, t)
      → None (don't commit) or a SwingCommitment to arm on the rod.
      Called every tick for each free controlled rod (proactive commitment).
      The swing connects only if the ball reaches a player during its active
      window; a mistimed swing whiffs and the held rod rigid-bounces.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

import config
from field import BallState, Field, Rod, SwingCommitment


# ---------------------------------------------------------------------------
# Module-level geometry helpers
# ---------------------------------------------------------------------------

def _reflect_through(
    ball_x: float, ball_y: float,
    wall_y: float,
    target_x: float, target_y: float,
) -> tuple[float, float]:
    """
    Mirror/ghost method: reflect target across wall_y, draw line from ball to
    ghost, return the intersection point on wall_y.
    """
    ghost_y = 2.0 * wall_y - target_y
    if abs(ghost_y - ball_y) < 1e-6:
        return ball_x, wall_y
    t = (wall_y - ball_y) / (ghost_y - ball_y)
    return ball_x + t * (target_x - ball_x), wall_y


def _best_wall_shot(
    rod: Rod, ball: BallState, field: Field,
    aim_x: float, aim_y: float,
) -> tuple[float, float]:
    """
    Return the aim point (on near or far side wall) that bounces toward
    (aim_x, aim_y). Picks the wall whose reflection gives a shot in bounds.
    Falls back to direct aim if neither wall produces a valid shot.
    """
    for wall_y in (0.0, field.width):
        ax, ay = _reflect_through(ball.x, ball.y, wall_y, aim_x, aim_y)
        if 0.0 <= ax <= field.depth:
            return ax, ay
    return aim_x, aim_y


def _best_player_deflection(
    rod: Rod, ball: BallState, field: Field,
    aim_x: float, aim_y: float,
) -> tuple[float, float]:
    """
    Aim at the edge of the nearest opponent player so the ball deflects toward
    (aim_x, aim_y). Uses the same mirror method with the player face as wall.
    Falls back to direct aim if no opponent player found.
    """
    team = rod.team
    if team == 0:
        opp = [r for r in field.rods if r.team != team and r.x > rod.x]
        opp.sort(key=lambda r: r.x)
    else:
        opp = [r for r in field.rods if r.team != team and r.x < rod.x]
        opp.sort(key=lambda r: -r.x)

    if not opp:
        return aim_x, aim_y

    nearest = opp[0]
    # Try both edges of the nearest player closest to ball.y
    best_pos = min(nearest.player_positions, key=lambda py: abs(py - ball.y))
    best: Optional[tuple[float, float]] = None
    for edge_y in (best_pos - nearest.width / 2, best_pos + nearest.width / 2):
        ax, ay = _reflect_through(ball.x, ball.y, edge_y, aim_x, aim_y)
        if 0.0 <= ax <= field.depth:
            best = ax, ay
            break
    return best if best is not None else (aim_x, aim_y)


def _project_ball_to_x(ball: BallState, target_x: float) -> Optional[float]:
    """
    Linear projection: return predicted ball.y when it reaches target_x.
    Returns None if ball is stationary or moving away from target_x.
    """
    if ball.vx == 0:
        return None
    t = (target_x - ball.x) / ball.vx
    if t < 0:
        return None
    return ball.y + ball.vy * t


def _predict_contact(ball: BallState, rod: Rod) -> Optional[tuple[float, float]]:
    """
    Predict when and where the ball reaches this rod's contact plane.

    The contact plane is the near x-face of the rod's players:
        rod.x ± (rod.thickness / 2 + config.BALL_RADIUS)
    (use the face the ball is approaching from).

    Returns
    -------
    (ttc, predicted_y) — time-to-contact in seconds and the ball's y at contact.
    None               — ball is moving away, will stop before reaching the rod,
                         or otherwise won't make contact.

    Notes for implementation (slice 4 — yours)
    -------------------------------------------
    * This is the *true* (skill-free) geometric estimate. Anticipation noise is
      added later in Strategy._commit_swing, not here.
    * Account for friction: the ball decelerates at config.FRICTION (cm/s²) along
      its travel direction, so a far/slow ball may never arrive — return None.
    * predicted_y can reuse the linear idea in _project_ball_to_x; optionally
      reflect off the side walls (0 .. field.width) for a y bounce, but a simple
      linear y estimate is a fine first cut.
    * Keep it cheap — this runs per controlled rod per tick.
    """
    # TODO(slice 4): implement the friction-aware TTC + predicted-y projection.
    raise NotImplementedError("_predict_contact: implement TTC projection (slice 4)")


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
    def choose_pos(
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
        ball: BallState,
        field: Field,
        t: float,
    ) -> Optional[SwingCommitment]:
        """
        Decide whether to commit a swing on this controlled rod right now.

        Called every tick for each free controlled rod (no pending swing, team
        not reaction-locked). To connect, the swing's active window must bracket
        the ball's actual arrival — so the decision hinges on predicting contact.

        Parameters
        ----------
        rod   : the controlled rod considering a swing.
        ball  : current ball state (position + velocity).
        field : full field (read-only).
        t     : current sim time (seconds) — the commit time.

        Returns
        -------
        None             — don't commit this tick.
        SwingCommitment  — arm this swing (typically built via _commit_swing).
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

    def _commit_swing(
        self,
        rod: Rod,
        ball: BallState,
        field: Field,
        t: float,
        aim_x: float,
        aim_y: float,
        intended_speed: float,
    ) -> Optional[SwingCommitment]:
        """
        Decide whether *now* is the moment to commit a swing aimed at (aim_x,
        aim_y), and if so build the SwingCommitment. Shared by every strategy —
        each strategy only supplies its aim point + intended speed.

        Algorithm (slice 5 — yours)
        ----------------------------
        1. pred = _predict_contact(ball, rod); if None → return None.
           Unpack (ttc_true, predicted_y).
        2. Bail if ttc_true > config.ANTICIPATION_MAX_HORIZON (too far to judge).
        3. Apply anticipation noise to the *estimate*:
               sigma   = config.ANTICIPATION_TTC_NOISE * ttc_true * (1 - rod.anticipation)
               ttc_est = ttc_true + Normal(0, sigma)
           (Lower anticipation → larger sigma → noisier estimate at long range.)
        4. Commit only if ttc_est lands in the lead band — i.e. the swing armed
           now would have its active window bracket the predicted arrival:
               backswing = config.SWING_DURATION * config.BACKSWING_RATIO
               if not (backswing <= ttc_est <= config.SWING_DURATION): return None
        5. Pick the contact player: the rod player whose y is nearest predicted_y;
           use that player's y as the _apply_hit origin (not a fixed player_idx).
        6. vel = self._apply_hit(rod, aim_x, aim_y, player_y, intended_speed);
           if None → return None.
        7. Return SwingCommitment(vx, vy,
               active_start = t + backswing,
               window_end   = t + config.SWING_DURATION).

        Note: timing uses the *true* ttc only via the band check on ttc_est;
        the actual hit/whiff is decided later by physics in simulate_point when
        the ball does (or doesn't) arrive inside [active_start, window_end].
        """
        # TODO(slice 5): implement anticipation-noise + lead-band commit logic.
        raise NotImplementedError("_commit_swing: implement commit logic (slice 5)")

    def _aim_forward(
        self, rod: Rod, ball: BallState, field: Field
    ) -> tuple[float, float]:
        """Return (aim_x, aim_y) targeting the nearest friendly rod ahead (toward opponent goal)."""
        team = rod.team
        if team == 0:
            ahead = [r for r in field.rods if r.team == team and r.x > rod.x]
            ahead.sort(key=lambda r: r.x)
        else:
            ahead = [r for r in field.rods if r.team == team and r.x < rod.x]
            ahead.sort(key=lambda r: -r.x)

        if ahead:
            t = ahead[0]
            return t.x, t.y_offset + field.width / 2
        # No friendly rod ahead — aim at goal
        goal = field.goal_for_attacker(team)
        return goal.x, (goal.y_min + goal.y_max) / 2

    def _aim_back(
        self, rod: Rod, ball: BallState, field: Field
    ) -> tuple[float, float]:
        """Return (aim_x, aim_y) targeting the nearest friendly rod behind (toward own goal)."""
        team = rod.team
        if team == 0:
            behind = [r for r in field.rods if r.team == team and r.x < rod.x]
            behind.sort(key=lambda r: -r.x)
        else:
            behind = [r for r in field.rods if r.team == team and r.x > rod.x]
            behind.sort(key=lambda r: r.x)

        if behind:
            t = behind[0]
            return t.x, t.y_offset + field.width / 2
        return self._aim_forward(rod, ball, field)

    def _aim_side(
        self, rod: Rod, player_idx: int, ball: BallState, field: Field
    ) -> tuple[float, float]:
        """Return (aim_x, aim_y) targeting the nearest other player on the same rod."""
        positions = rod.player_positions
        if len(positions) <= 1:
            # No other player — fall back to forward aim
            return self._aim_forward(rod, ball, field)
        player_y = positions[player_idx]
        other = min(
            (py for i, py in enumerate(positions) if i != player_idx),
            key=lambda py: abs(py - player_y),
        )
        return rod.x, other

    def _find_gap_target(
        self, rod: Rod, ball: BallState, field: Field
    ) -> tuple[float, float]:
        """Find (aim_x, aim_y) through the largest gap in the nearest opponent rod."""
        team = rod.team
        goal = field.goal_for_attacker(team)

        if team == 0:
            opp_rods = [r for r in field.rods if r.team != team and r.x > rod.x]
            opp_rods.sort(key=lambda r: r.x)
        else:
            opp_rods = [r for r in field.rods if r.team != team and r.x < rod.x]
            opp_rods.sort(key=lambda r: -r.x)

        if not opp_rods:
            return goal.x, (goal.y_min + goal.y_max) / 2

        nearest = opp_rods[0]
        positions = sorted(nearest.player_positions)
        reach = nearest.width / 2

        gaps: list[tuple[float, float]] = []
        if positions[0] - reach > 0:
            gaps.append((0.0, positions[0] - reach))
        for i in range(len(positions) - 1):
            lo = positions[i] + reach
            hi = positions[i + 1] - reach
            if hi > lo:
                gaps.append((lo, hi))
        if positions[-1] + reach < field.width:
            gaps.append((positions[-1] + reach, field.width))

        if gaps:
            best = max(gaps, key=lambda g: g[1] - g[0])
            return nearest.x, (best[0] + best[1]) / 2

        return goal.x, (goal.y_min + goal.y_max) / 2


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

    def choose_pos(
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

    def choose_pos(
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

    def choose_pos(
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

    def choose_pos(
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

    def choose_pos(
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

    def choose_pos(
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