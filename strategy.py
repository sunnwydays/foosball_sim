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


def _open_gaps(opp_rod: Rod, field: Field) -> list[tuple[float, float]]:
    """
    Open y-intervals on opp_rod's plane that the ball can pass through.
    Blocked intervals are player center +/- (player half-width + ball radius),
    since the ball's edge must clear the player's edge.
    """
    positions = sorted(opp_rod.player_positions)
    reach = opp_rod.width / 2 + config.BALL_RADIUS

    gaps: list[tuple[float, float]] = []
    if positions[0] - reach > 0.0:
        gaps.append((0.0, positions[0] - reach))
    for i in range(len(positions) - 1):
        lo = positions[i] + reach
        hi = positions[i + 1] - reach
        if hi > lo:
            gaps.append((lo, hi))
    if positions[-1] + reach < field.width:
        gaps.append((positions[-1] + reach, field.width))
    return gaps


def _aim_through_gap(
    origin_x: float, origin_y: float,
    opp_rod: Rod, goal,
    field: Field,
) -> Optional[tuple[float, float]]:
    """
    Aim point (on the goal plane) for a straight shot from (origin_x, origin_y)
    that passes through an open gap in opp_rod AND enters the goal mouth.

    Projects the goal mouth back onto the defender's plane, intersects that
    window with the open gaps, and aims through the center of the widest
    overlap. Returns None when no gap lines up with the goal.
    """
    span = goal.x - origin_x
    if abs(span) < 1e-6:
        return None
    frac = (opp_rod.x - origin_x) / span
    if not (0.0 < frac < 1.0):
        return None

    win_lo = origin_y + (goal.y_min - origin_y) * frac
    win_hi = origin_y + (goal.y_max - origin_y) * frac

    best: Optional[tuple[float, float]] = None
    best_width = 0.0
    for lo, hi in _open_gaps(opp_rod, field):
        o_lo = max(lo, win_lo)
        o_hi = min(hi, win_hi)
        if o_hi - o_lo > best_width:
            best_width = o_hi - o_lo
            best = (o_lo, o_hi)

    if best is None:
        return None
    gap_y = (best[0] + best[1]) / 2.0
    return goal.x, origin_y + (gap_y - origin_y) / frac


def _fold_into_field(y: float, width: float) -> float:
    """Triangle-wave fold: reflect y into [0, width] as the ball bounces off side walls."""
    if width <= 0:
        return y
    period = 2.0 * width
    m = y % period
    return m if m <= width else period - m


def _time_to_reach_x(x0: float, vx: float, speed: float, target_x: float) -> Optional[float]:
    """
    Smallest non-negative time at which a ball at x0 with x-velocity vx (total
    speed `speed`) reaches target_x under isotropic friction, or None if it
    stops first / never reaches it.

    x(t) = x0 + vx*t + a*t^2,  a = -(vx / speed) * FRICTION / 2  (opposes vx).
    Valid only until the ball stops, at t_stop = speed / FRICTION.
    """
    a = -(vx / speed) * config.FRICTION / 2.0
    b = vx
    c = x0 - target_x

    discriminant = b*b - 4*a*c
    if discriminant < 0:
        return None

    sqrt_term = math.sqrt(discriminant)
    t_stop = speed / config.FRICTION
    candidates = [
        t for t in ((-b + sqrt_term) / (2 * a), (-b - sqrt_term) / (2 * a))
        if 0.0 <= t <= t_stop
    ]
    return min(candidates) if candidates else None


# Max end-wall bounces to follow when predicting contact (ball decelerates, so
# this terminates anyway; the cap is a safety bound).
_MAX_PREDICT_BOUNCES = 4


def _predict_contact(ball: BallState, rod: Rod) -> Optional[tuple[float, float]]:
    """
    Predict when and where the ball reaches this rod's contact face, following
    end-wall (x) bounces so a rod can anticipate the ball's return off the back
    wall. Side-wall (y) bounces are not folded here — callers apply
    `_fold_into_field` to the returned y.

    Contact face: the near x-face of the player block (including ball radius),
    recomputed from the ball's current x-direction after each bounce:
        face_x = rod.x - copysign(thickness/2 + BALL_RADIUS, vx)

    End walls are at x=BALL_RADIUS and x=FIELD_DEPTH-BALL_RADIUS; the goal
    opening is treated as a solid wall (always reflect in x). Friction is
    isotropic (matches physics.py): total speed decelerates at FRICTION, so x
    and y deceleration are proportional and y integration is unaffected by
    x-reflections — predicted_y uses the full ttc with the original vy.

    Returns
    -------
    (ttc, predicted_y) — time-to-contact and ball y at contact (unfolded).
    None               — ball has no x-motion, or stops before reaching the rod.
    """
    # No x-motion: ball won't travel toward the rod. Check static overlap.
    if abs(ball.vx) <= config.STOP_THRESHOLD:
        if abs(ball.x - rod.x) <= rod.rod_x_reach + rod.thickness / 2 + config.BALL_RADIUS:
            return (0.0, ball.y)
        return None

    reach = rod.thickness / 2 + config.BALL_RADIUS

    # Already within the contact band while moving: immediate contact.
    if abs(ball.x - rod.x) <= reach:
        return (0.0, ball.y)

    left_wall  = config.BALL_RADIUS
    right_wall = config.FIELD_DEPTH - config.BALL_RADIUS

    # Walk x-segments, reflecting off end walls, until the ball reaches the rod
    # face or stops. y integration uses the original ball state over total ttc.
    x        = ball.x
    vx       = ball.vx
    speed    = ball.speed
    elapsed  = 0.0

    for _ in range(_MAX_PREDICT_BOUNCES + 1):
        if abs(vx) <= config.STOP_THRESHOLD:
            return None

        face_x = rod.x - math.copysign(reach, vx)
        t_face = _time_to_reach_x(x, vx, speed, face_x)

        wall   = right_wall if vx > 0 else left_wall
        t_wall = _time_to_reach_x(x, vx, speed, wall)

        # Reach the rod this segment (before any bounce) → that's the contact.
        if t_face is not None and (t_wall is None or t_face <= t_wall):
            ttc = elapsed + t_face
            a_y = -(ball.vy / ball.speed) * config.FRICTION / 2.0
            predicted_y = ball.y + ball.vy * ttc + a_y * ttc * ttc
            return (ttc, predicted_y)

        # Otherwise advance to the wall and reflect, if the ball gets there.
        if t_wall is None:
            return None  # ball stops before the wall and never reaches the rod

        new_speed = speed - config.FRICTION * t_wall
        if new_speed <= config.STOP_THRESHOLD:
            return None  # effectively stops at the wall

        # vx scales with the decayed speed, then flips at the wall.
        vx      = -vx * (new_speed / speed)
        speed   = new_speed
        x       = wall
        elapsed += t_wall

    return None

class Strategy(ABC):

    def choose_hands(
        self,
        team_rods: list[tuple[int, Rod]],
        ball: BallState,
        field: Field,
        n_hands: int,
    ) -> set[int]:
        """
        Pick which rods (up to n_hands) this team controls this tick.

        Default: grab the rod(s) the ball will next reach, so SWITCH_DELAY
        expires before contact. Falls back to nearest-x when the ball is slow,
        stopped, or moving away from every rod.

        Returns a set of global rod indices.
        """
        incoming = []                    # (ttc, idx) for rods in the ball's path
        for idx, rod in team_rods:
            contact = _predict_contact(ball, rod)
            if contact is not None:
                incoming.append((contact[0], idx))
        chosen = [idx for _, idx in sorted(incoming)[:n_hands]]
        if len(chosen) < n_hands:        # fill remaining slots by nearest-x
            for idx, _ in sorted(team_rods, key=lambda ir: abs(ir[1].x - ball.x)):
                if idx not in chosen:
                    chosen.append(idx)
                if len(chosen) >= n_hands:
                    break
        return set(chosen)

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
        """
        contact = _predict_contact(ball, rod)
        if contact is None:
            return None
        ttc_true, predicted_y = contact
        predicted_y = _fold_into_field(predicted_y, field.width)

        # Ball is already at the rod so arm an immediate hit (no lead band needed)
        if ttc_true <= 0.0:
            vel = self._apply_hit(rod, aim_x, aim_y, predicted_y, intended_speed)
            if vel is None:
                return None
            vx, vy = vel
            return SwingCommitment(vx=vx, vy=vy, active_start=t, window_end=t + config.DT)

        # Bail if ball too far to predict reliably
        if ttc_true > config.ANTICIPATION_MAX_HORIZON:
            return None

        # Apply anticipation noise
        sigma = config.ANTICIPATION_TTC_NOISE * ttc_true * (1 - rod.anticipation)
        ttc_est = ttc_true + (float(np.random.normal(0, sigma)) if sigma > 0 else 0.0)

        # Commit only if estimated arrival falls in the swing lead band
        backswing = config.SWING_DURATION * config.BACKSWING_RATIO
        if not (backswing <= ttc_est <= config.SWING_DURATION):
            return None

        # Player whose y is nearest the predicted contact point
        nearest_player_y = min(rod.player_positions, key=lambda py: abs(py - predicted_y))

        vel = self._apply_hit(rod, aim_x, aim_y, nearest_player_y, intended_speed)
        if vel is None:
            return None

        vx, vy = vel
        return SwingCommitment(
            vx=vx,
            vy=vy,
            active_start=t + backswing,
            window_end=t + config.SWING_DURATION,
        )

    def _cover_offset(self, rod: Rod, target_y: float) -> float:
        """
        Return the y_offset target that puts the best-covering player directly
        on absolute y `target_y`, respecting the slide clamp.

        Centering the rod (target_y - field.width / 2) is wrong in two cases:
        when the required offset exceeds the slide range it clamps into a
        coverage gap where NO player can reach the ball (possession stall),
        and on even-player rods it centers the gap between players on the
        ball instead of a player. Ties on coverage error are broken by least
        travel from the current offset.
        """
        lo, hi = rod.slide_range
        best_off = 0.0
        best_key = None
        for bp in rod._base_positions:
            off = max(lo, min(hi, target_y - bp))
            key = (abs(bp + off - target_y), abs(off - rod.y_offset))
            if best_key is None or key < best_key:
                best_key, best_off = key, off
        return best_off

    def _predicted_y(self, rod: Rod, ball: BallState, field: Field) -> float:
        """Absolute y where ball is predicted to reach this rod's x, folded for
        side-wall bounces. Falls back to ball.y when no clean intercept."""
        contact = _predict_contact(ball, rod)
        if contact is None:
            return ball.y
        _, predicted_y = contact
        return _fold_into_field(predicted_y, field.width)

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
        self, rod: Rod, ball: BallState, field: Field
    ) -> tuple[float, float]:
        """Return (aim_x, aim_y) targeting the nearest other player on the same rod."""
        positions = rod.player_positions
        if len(positions) <= 1:
            return self._aim_forward(rod, ball, field)
        nearest_y = min(positions, key=lambda py: abs(py - ball.y))
        other = min(
            (py for py in positions if py != nearest_y),
            key=lambda py: abs(py - nearest_y),
        )
        return rod.x, other

    def _find_gap_target(
        self, rod: Rod, ball: BallState, field: Field, skip_up: bool = False,
    ) -> tuple[float, float]:
        """
        Find (aim_x, aim_y) that threads a gap in the nearest opponent rod and
        continues into the goal mouth. Falls back to the largest gap (advances
        the ball) when no gap lines up with the goal, then to goal center when
        the defender plane is fully blocked.

        skip_up=True ignores opponent rods that are flipped up (can't block).
        """
        team = rod.team
        goal = field.goal_for_attacker(team)

        if team == 0:
            opp_rods = [r for r in field.rods if r.team != team and r.x > rod.x
                        and not (skip_up and r.up)]
            opp_rods.sort(key=lambda r: r.x)
        else:
            opp_rods = [r for r in field.rods if r.team != team and r.x < rod.x
                        and not (skip_up and r.up)]
            opp_rods.sort(key=lambda r: -r.x)

        if not opp_rods:
            return goal.x, (goal.y_min + goal.y_max) / 2

        nearest = opp_rods[0]

        # Shot origin: the player that will actually take the shot, matching
        # the nearest-player choice made by _commit_swing.
        predicted_y = self._predicted_y(rod, ball, field)
        origin_y = min(rod.player_positions, key=lambda py: abs(py - predicted_y))

        aim = _aim_through_gap(rod.x, origin_y, nearest, goal, field)
        if aim is not None:
            return aim

        gaps = _open_gaps(nearest, field)
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

    def choose_pos(
        self,
        controlled_rods: list[tuple[int, Rod]],
        ball: BallState,
        field: Field,
    ) -> dict[int, tuple[float, float, bool]]:
        targets = {}
        for rod_idx, rod in controlled_rods:
            target_y = self._cover_offset(rod, self._predicted_y(rod, ball, field))
            targets[rod_idx] = (target_y, 0.0, False)
        return targets

    def choose_hit(
        self,
        rod: Rod,
        ball: BallState,
        field: Field,
        t: float,
    ) -> Optional[SwingCommitment]:
        goal = field.goal_for_attacker(rod.team)
        aim_x = goal.x
        aim_y = (goal.y_min + goal.y_max) / 2
        return self._commit_swing(rod, ball, field, t, aim_x, aim_y, config.HIT_SPEED)

# ---------------------------------------------------------------------------
# AimAtGap — scans for defensive gaps before hitting
# ---------------------------------------------------------------------------

class AimAtGap(Strategy):
    """
    Like SmackBall but aims through the largest gap in the nearest opponent rod
    between the ball and the goal.
    """

    def choose_pos(
        self,
        controlled_rods: list[tuple[int, Rod]],
        ball: BallState,
        field: Field,
    ) -> dict[int, tuple[float, float, bool]]:
        targets = {}
        for rod_idx, rod in controlled_rods:
            target_y = self._cover_offset(rod, self._predicted_y(rod, ball, field))
            targets[rod_idx] = (target_y, 0.0, False)
        return targets

    def choose_hit(
        self,
        rod: Rod,
        ball: BallState,
        field: Field,
        t: float,
    ) -> Optional[SwingCommitment]:
        aim_x, aim_y = self._find_gap_target(rod, ball, field)
        return self._commit_swing(rod, ball, field, t, aim_x, aim_y, config.HIT_SPEED)

# ---------------------------------------------------------------------------
# HardOffense — trying to hit into the goal, remove obstacles
# ---------------------------------------------------------------------------

class HardOffense(Strategy):
    """
    Like SmackBall but aims through the largest gap in the nearest opponent rod
    between the ball and the goal.
    """

    def choose_pos(
        self,
        controlled_rods: list[tuple[int, Rod]],
        ball: BallState,
        field: Field,
    ) -> dict[int, tuple[float, float, bool]]:
        targets = {}
        for rod_idx, rod in controlled_rods:
            target_y = self._cover_offset(rod, self._predicted_y(rod, ball, field))

            # Flip rod up if ball is moving away from opponent's goal past this rod
            up = False
            if rod.team == 0:
                up = ball.vx < -config.STOP_THRESHOLD and rod.x > ball.x
            else:
                up = ball.vx > config.STOP_THRESHOLD and rod.x < ball.x

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
        ball: BallState,
        field: Field,
        t: float,
    ) -> Optional[SwingCommitment]:
        goal = field.goal_for_attacker(rod.team)
        aim_x = goal.x
        goal_cy = (goal.y_min + goal.y_max) / 2
        goal_half = (goal.y_max - goal.y_min) / 2
        aim_y = float(np.clip(np.random.normal(goal_cy, goal_half / 2), goal.y_min, goal.y_max))
        return self._commit_swing(rod, ball, field, t, aim_x, aim_y, config.HIT_SPEED * 1.5)

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

    def choose_pos(
        self,
        controlled_rods: list[tuple[int, Rod]],
        ball: BallState,
        field: Field,
    ) -> dict[int, tuple[float, float, bool]]:
        targets = {}
        for rod_idx, rod in controlled_rods:
            target_y = self._cover_offset(rod, self._predicted_y(rod, ball, field))
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
        ball: BallState,
        field: Field,
        t: float,
    ) -> Optional[SwingCommitment]:
        goal = field.goal_for_attacker(rod.team)
        aim_x = goal.x
        aim_y = (goal.y_min + goal.y_max) / 2
        return self._commit_swing(rod, ball, field, t, aim_x, aim_y, config.HIT_SPEED)

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

    def choose_pos(
        self,
        controlled_rods: list[tuple[int, Rod]],
        ball: BallState,
        field: Field,
    ) -> dict[int, tuple[float, float, bool]]:
        targets = {}
        for rod_idx, rod in controlled_rods:
            target_y = self._cover_offset(rod, self._predicted_y(rod, ball, field))

            # Flip up controlled rod if ball is retreating past it
            up = False
            if rod.team == 0:
                up = ball.vx < -config.STOP_THRESHOLD and rod.x > ball.x
            else:
                up = ball.vx > config.STOP_THRESHOLD and rod.x < ball.x

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
        ball: BallState,
        field: Field,
        t: float,
    ) -> Optional[SwingCommitment]:
        aim_x, aim_y = self._find_gap_target(rod, ball, field, skip_up=True)
        return self._commit_swing(rod, ball, field, t, aim_x, aim_y, config.HIT_SPEED)


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

    def choose_pos(
        self,
        controlled_rods: list[tuple[int, Rod]],
        ball: BallState,
        field: Field,
    ) -> dict[int, tuple[float, float, bool]]:
        targets = {}
        for rod_idx, rod in controlled_rods:
            target_y = self._cover_offset(rod, self._predicted_y(rod, ball, field))
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
        ball: BallState,
        field: Field,
        t: float,
    ) -> Optional[SwingCommitment]:
        aim_x, aim_y = self._find_gap_target(rod, ball, field, skip_up=True)
        return self._commit_swing(rod, ball, field, t, aim_x, aim_y, config.HIT_SPEED)