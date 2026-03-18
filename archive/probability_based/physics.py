"""
physics.py — Ball trajectory tracing via ray-casting.

How it works
------------
A shot sends the ball in a straight line from (start_x, start_y) at angle θ.
We process the path segment-by-segment, where each segment ends at either:

  1. A SIDE WALL (y=0 or y=FIELD_DEPTH)
       → ball reflects: y-component of velocity flips  (angle_new = atan2(-sin θ, cos θ))
       → new segment begins from the wall at the updated angle.

  2. A PLAYER BOUNDING BOX (2D AABB intersection)
       → if the ray hits any player's x/y bounding box → POSSESSION TRANSFER
       → contact position is the actual ray entry point into the box.

  3. AN END WALL / GOAL (x=0 or x=FIELD_WIDTH)
       → if y lands inside the goal opening → GOAL
       → otherwise → DEAD BALL (hit the wall outside the goal).

Each segment is bounded by min(t_wall, t_end) so a wall bounce always happens
before we check for player hits beyond it.  Players are processed by first
contact time, so the nearest player-box hit wins.

possessing_rod_idx
------------------
The rod that just shot the ball is skipped for the first segment only.
After a wall bounce that rod becomes eligible again (the ball genuinely came
back and could be trapped).

Coordinate convention
---------------------
angle=0   → rightward  (+x)   Team 0 attacks in this direction
angle=π/2 → upward     (+y)
angle=π   → leftward   (-x)   Team 1 attacks in this direction
"""

import math
from field import Field, TraceResult
import config

_EPS = 1e-9   # numerical guard


def _ray_aabb_t(
    ox: float, oy: float,
    dx: float, dy: float,
    xmin: float, xmax: float,
    ymin: float, ymax: float,
) -> float | None:
    """
    Ray–AABB intersection (slab method).

    Returns the t of first contact (ray entry into the box), or None if the
    ray misses.  Ray: P(t) = (ox + t·dx, oy + t·dy).

    A negative t means the box is behind the ray origin; callers should
    ignore hits with t ≤ _EPS.
    """
    INF = math.inf

    # --- x slab ---
    if abs(dx) > _EPS:
        tx1 = (xmin - ox) / dx
        tx2 = (xmax - ox) / dx
        t_enter_x = min(tx1, tx2)
        t_exit_x  = max(tx1, tx2)
    elif xmin <= ox <= xmax:
        t_enter_x, t_exit_x = -INF, INF   # ray parallel, inside slab
    else:
        return None                         # ray parallel, outside slab

    # --- y slab ---
    if abs(dy) > _EPS:
        ty1 = (ymin - oy) / dy
        ty2 = (ymax - oy) / dy
        t_enter_y = min(ty1, ty2)
        t_exit_y  = max(ty1, ty2)
    elif ymin <= oy <= ymax:
        t_enter_y, t_exit_y = -INF, INF
    else:
        return None

    t_enter = max(t_enter_x, t_enter_y)
    t_exit  = min(t_exit_x,  t_exit_y)

    if t_enter > t_exit:
        return None   # slabs don't overlap — miss

    return t_enter


def trace_ball(
    start_x: float,
    start_y: float,
    angle:   float,
    field:   Field,
    possessing_rod_idx: int = -1,
) -> TraceResult:
    """
    Trace the ball from (start_x, start_y) travelling at `angle` (radians).

    Parameters
    ----------
    start_x, start_y    : ball's starting position.
    angle               : shot direction in radians.
    field               : Field instance with all rods at their current offsets.
    possessing_rod_idx  : rod that just shot — skipped for the first segment so
                          the ball doesn't immediately re-intercept its origin.

    Returns
    -------
    TraceResult with kind ∈ {'goal', 'possession', 'dead'}.
    """
    x, y = start_x, start_y
    skip_rod = possessing_rod_idx   # cleared after the first wall bounce

    for _bounce in range(config.MAX_WALL_BOUNCES):

        dx = math.cos(angle)
        dy = math.sin(angle)

        # ------------------------------------------------------------------
        # 1. Distance to the next side wall (y-boundaries)
        # ------------------------------------------------------------------
        if dy > _EPS:
            t_wall = (field.depth - y) / dy     # hitting the TOP wall
        elif dy < -_EPS:
            t_wall = -y / dy                     # hitting the BOTTOM wall
        else:
            t_wall = math.inf                    # moving horizontally — no wall hit

        # ------------------------------------------------------------------
        # 2. Distance to the end walls / goal lines (x-boundaries)
        # ------------------------------------------------------------------
        if dx > _EPS:
            t_end = (field.width - x) / dx      # heading RIGHT → right goal line
        elif dx < -_EPS:
            t_end = -x / dx                      # heading LEFT  → left goal line
        else:
            t_end = math.inf                     # moving vertically — no end wall hit

        # Only check for player hits within this segment
        t_limit = min(t_wall, t_end)

        # ------------------------------------------------------------------
        # 3. Find all player-bounding-box intersections within this segment
        # ------------------------------------------------------------------
        player_hits: list[tuple[float, int, float, float]] = []  # (t, rod_idx, cx, cy)

        for rod_idx, rod in enumerate(field.rods):
            if rod_idx == skip_rod:
                continue   # skip the shooting rod for this segment

            for player_y in rod.player_positions:
                t = _ray_aabb_t(
                    x, y, dx, dy,
                    rod.x - config.PLAYER_X_REACH, rod.x + config.PLAYER_X_REACH,
                    player_y  - rod.y_reach,        player_y  + rod.y_reach,
                )
                if t is not None and _EPS < t < t_limit:
                    player_hits.append((t, rod_idx, x + t * dx, y + t * dy))

        if player_hits:
            player_hits.sort(key=lambda h: h[0])
            _, rod_idx, cx, cy = player_hits[0]
            return TraceResult(kind='possession', rod_idx=rod_idx, x=cx, y=cy)

        # ------------------------------------------------------------------
        # 4. No player hit: advance to whichever boundary comes first
        # ------------------------------------------------------------------
        if t_end <= t_wall:
            # Reached the end wall / goal line
            y_end = y + t_end * dy

            goal = field.right_goal if dx > 0 else field.left_goal

            if goal.contains(y_end):
                return TraceResult(kind='goal', scoring_team=goal.scoring_team)
            else:
                return TraceResult(kind='dead')   # hit wall outside goal opening

        else:
            # Wall bounce: advance to the side wall and flip the y-velocity
            x = x + t_wall * dx
            y = field.depth if dy > 0 else 0.0
            y = max(0.0, min(field.depth, y))    # clamp floating-point drift

            angle    = math.atan2(-dy, dx)
            skip_rod = -1   # possessing rod is eligible again after a bounce

    # Exceeded MAX_WALL_BOUNCES — treat as dead ball
    return TraceResult(kind='dead')
