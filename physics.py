"""
physics.py — Ball trajectory tracing via ray-casting.

How it works
------------
A shot sends the ball in a straight line from (start_x, start_y) at angle θ.
We process the path segment-by-segment, where each segment ends at either:

  1. A SIDE WALL (y=0 or y=FIELD_HEIGHT)
       → ball reflects: y-component of velocity flips  (angle_new = atan2(-sin θ, cos θ))
       → new segment begins from the wall at the updated angle.

  2. A ROD INTERSECTION (ball's x-path crosses a rod's x-position)
       → if a player's hitbox covers the ball's y at that x → POSSESSION TRANSFER
       → if no player covers that y → ball passes through (no obstruction).

  3. AN END WALL / GOAL (x=0 or x=FIELD_WIDTH)
       → if y lands inside the goal opening → GOAL
       → otherwise → DEAD BALL (hit the wall outside the goal).

Within each segment we only check rods whose t_rod < min(t_wall, t_end),
so a wall bounce always happens before we check rods beyond it.
Rods are processed in order of distance (closest first), so the first
player-hitbox hit wins.

Coordinate convention
---------------------
angle=0   → rightward  (+x)   Team 0 attacks in this direction
angle=π/2 → upward     (+y)
angle=π   → leftward   (-x)   Team 1 attacks in this direction
"""

import math
from field import Field, TraceResult
import config

_EPS = 1e-9   # numerical guard — avoids re-intercepting the origin rod


def trace_ball(
    start_x: float,
    start_y: float,
    angle:   float,
    field:   Field,
) -> TraceResult:
    """
    Trace the ball from (start_x, start_y) travelling at `angle` (radians).

    Parameters
    ----------
    start_x, start_y : ball's starting position (should equal the possessing
                        rod's x and the player's current y).
    angle            : shot direction in radians.
    field            : Field instance with all rods at their current y_offsets.

    Returns
    -------
    TraceResult with kind ∈ {'goal', 'possession', 'dead'}.
    """
    x, y = start_x, start_y

    for _bounce in range(config.MAX_WALL_BOUNCES):

        dx = math.cos(angle)
        dy = math.sin(angle)

        # ------------------------------------------------------------------
        # 1. Distance to the next side wall (y-boundaries)
        # ------------------------------------------------------------------
        if dy > _EPS:
            t_wall = (field.height - y) / dy        # hitting the TOP wall
        elif dy < -_EPS:
            t_wall = -y / dy                         # hitting the BOTTOM wall
        else:
            t_wall = math.inf                        # moving horizontally — no wall hit

        # ------------------------------------------------------------------
        # 2. Distance to the end walls / goal lines (x-boundaries)
        # ------------------------------------------------------------------
        if dx > _EPS:
            t_end = (field.width - x) / dx          # heading RIGHT → right goal line
        elif dx < -_EPS:
            t_end = -x / dx                          # heading LEFT  → left goal line
        else:
            t_end = math.inf                         # moving vertically — no end wall hit

        # Only look at rods within this segment (before any wall or end wall)
        t_limit = min(t_wall, t_end)

        # ------------------------------------------------------------------
        # 3. Find all rod intersections within this segment
        # ------------------------------------------------------------------
        rod_hits: list[tuple[float, int, float]] = []   # (t, rod_idx, y_at_rod)

        if abs(dx) > _EPS:   # ball must have x-component to cross a rod
            for rod_idx, rod in enumerate(field.rods):
                t_rod = (rod.x - x) / dx

                # Only rods that are AHEAD (positive t) and within this segment
                if _EPS < t_rod < t_limit:
                    y_at_rod = y + t_rod * dy
                    rod_hits.append((t_rod, rod_idx, y_at_rod))

        # Process rods closest-first
        rod_hits.sort(key=lambda h: h[0])

        for t_rod, rod_idx, y_at_rod in rod_hits:
            rod = field.rods[rod_idx]
            if rod.player_at_y(y_at_rod) is not None:
                # A player's hitbox covers this y → possession transfer
                # Snap to the player's center y for a clean next-shot origin
                player_idx = rod.player_at_y(y_at_rod)
                player_y   = rod.player_positions[player_idx]
                return TraceResult(
                    kind    = 'possession',
                    rod_idx = rod_idx,
                    x       = rod.x,
                    y       = player_y,
                )
            # No player at that y — ball passes through the rod's column

        # ------------------------------------------------------------------
        # 4. No rod interception: advance to whichever boundary comes first
        # ------------------------------------------------------------------
        if t_end <= t_wall:
            # Reached the end wall / goal line
            y_end = y + t_end * dy

            if dx > 0:
                goal = field.right_goal   # Team 0 scored if it goes in
            else:
                goal = field.left_goal    # Team 1 scored if it goes in

            if goal.contains(y_end):
                return TraceResult(kind='goal', scoring_team=goal.scoring_team)
            else:
                return TraceResult(kind='dead')   # hit wall outside goal opening

        else:
            # Wall bounce: advance to the side wall and flip the y-velocity
            x = x + t_wall * dx
            y = field.height if dy > 0 else 0.0
            y = max(0.0, min(field.height, y))    # clamp floating-point drift

            # Reflect: keep x-component, negate y-component
            # atan2(-sin θ, cos θ) = -θ  for a horizontal-wall reflection
            angle = math.atan2(-dy, dx)

    # Exceeded MAX_WALL_BOUNCES — treat as dead ball
    return TraceResult(kind='dead')
