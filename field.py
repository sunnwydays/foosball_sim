"""
field.py — Core data structures for the foosball table.

Classes
-------
Rod        — A single foosball rod (slides in y, rotates modelled as x-slide).
Goal       — A goal opening on one end wall.
BallState  — The ball's position and velocity.
TeamState  — Per-team mutable state: active hands, reaction timer.
Field      — The complete table: rods, goals, dimensions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field as dc_field
from typing import Optional

import config


# ---------------------------------------------------------------------------
# Rod
# ---------------------------------------------------------------------------

class Rod:
    """
    A foosball rod.

    Physical layout
    ---------------
    - Base position at _base_x on the field.
    - Slides along the y-axis (all players move together).
    - x_offset models rotation (±rod_x_reach from base).
    - Players are evenly spaced vertically; their default centers are at
      (i+1) * (FIELD_WIDTH / (n_players + 1)) for i in 0..n_players-1.

    Continuous movement
    -------------------
    target_y  : where the rod is trying to slide to (y_offset target)
    vy        : current y-velocity (moves toward target_y at up to MOVEMENT_SPEED)
    switch_timer : seconds remaining before this rod responds to commands
                   (set when a hand switches to this rod)

    Skill parameters
    ----------------
    accuracy          : [0, 1] — controls shot angle std dev.
    power_consistency : [0, 1] — controls shot speed std dev.
    """

    def __init__(
        self,
        x: float,
        team: int,
        n_players: int,
        accuracy: float = 0.5,
        power_consistency: float = 0.5,
        width: float = config.PLAYER_WIDTH,
        thickness: float = config.PLAYER_THICKNESS,
        rod_x_reach: float = config.ROD_X_REACH,
    ):
        self._base_x     = x
        self.team        = team
        self.n_players   = n_players
        self.accuracy          = accuracy
        self.power_consistency = power_consistency
        self.width      = width
        self.thickness   = thickness
        self.rod_x_reach = rod_x_reach

        # --- Mutable positional state ---
        self.y_offset    = 0.0   # current y slide
        self.x_offset    = 0.0   # current x slide (rotation)

        # --- Continuous movement state ---
        self.target_y    = 0.0   # desired y_offset (rod moves toward this)
        self.vy          = 0.0   # current y-velocity (cm/s)

        # --- Control state ---
        self.controlled    = False  # is a hand currently on this rod?
        self.switch_timer  = 0.0    # seconds until rod responds after hand switch

        # --- Bounds ---
        # x slide: rod can move ±rod_x_reach from origin (models rotation)
        self._x_slide_min = -rod_x_reach
        self._x_slide_max =  rod_x_reach

        # Default player centers (equally spaced, centered on field depth)
        spacing = config.FIELD_WIDTH / (n_players + 1)
        self._base_positions: list[float] = [
            (i + 1) * spacing for i in range(n_players)
        ]

        # Valid y offset range: outermost players must stay within [0, FIELD_WIDTH]
        self._slide_min = -self._base_positions[0]  + width / 2
        self._slide_max =  config.FIELD_WIDTH - self._base_positions[-1] - width / 2

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def x(self) -> float:
        """Current x-coordinate of the rod (base position + x_offset)."""
        return self._base_x + self.x_offset

    @property
    def player_positions(self) -> list[float]:
        """Current y-coordinates of all players on this rod."""
        return [bp + self.y_offset for bp in self._base_positions]

    @property
    def slide_range(self) -> tuple[float, float]:
        """(min_offset, max_offset) — valid y sliding range."""
        return self._slide_min, self._slide_max

    @property
    def angle_std(self) -> float:
        """Shot angle std dev (radians) derived from accuracy."""
        return (
            config.MAX_ANGLE_STD
            + self.accuracy * (config.MIN_ANGLE_STD - config.MAX_ANGLE_STD)
        )

    @property
    def speed_std(self) -> float:
        """Speed std dev (cm/s) derived from power_consistency."""
        return (
            config.MAX_SPEED_STD
            + self.power_consistency * (config.MIN_SPEED_STD - config.MAX_SPEED_STD)
        )

    # ------------------------------------------------------------------
    # Methods
    # ------------------------------------------------------------------

    def set_offset(self, offset: float) -> None:
        """Slide rod to y `offset`, clamping to valid range."""
        self.y_offset = max(self._slide_min, min(self._slide_max, offset))

    def set_x_offset(self, offset: float) -> None:
        """Slide rod to x `offset`, clamping within rotation bounds."""
        self.x_offset = max(self._x_slide_min, min(self._x_slide_max, offset))

    def player_in_box(self, x: float, y: float) -> Optional[int]:
        """
        Return the index of the player whose 2D bounding box covers (x, y),
        or None.
        """
        for i, py in enumerate(self.player_positions):
            if abs(x - self.x) <= self.thickness / 2 and abs(y - py) <= self.width / 2:
                return i
        return None

    def step_movement(self, dt: float) -> None:
        """
        Move the rod toward target_y for one tick.

        If the rod is not controlled or its switch_timer is active,
        it stops (vy → 0) and stays in place.
        """
        # Tick down switch timer
        if self.switch_timer > 0:
            self.switch_timer = max(0.0, self.switch_timer - dt)

        # Can't move if not controlled or still switching
        if not self.controlled or self.switch_timer > 0:
            self.vy = 0.0
            return

        # Move toward target_y at up to MOVEMENT_SPEED
        diff = self.target_y - self.y_offset
        if abs(diff) < 1e-6:
            self.vy = 0.0
            return

        direction = 1.0 if diff > 0 else -1.0
        max_step = config.MOVEMENT_SPEED * dt
        step = min(abs(diff), max_step)

        self.vy = direction * config.MOVEMENT_SPEED
        self.set_offset(self.y_offset + direction * step)

    def reset(self) -> None:
        """Reset all mutable state to defaults."""
        self.y_offset     = 0.0
        self.x_offset     = 0.0
        self.target_y     = 0.0
        self.vy           = 0.0
        self.controlled   = False
        self.switch_timer = 0.0

    def __repr__(self) -> str:
        ctrl = "H" if self.controlled else "-"
        return (
            f"Rod(x={self.x:.1f}, team={self.team}, n={self.n_players}, "
            f"y_off={self.y_offset:.1f}, ctrl={ctrl})"
        )


# ---------------------------------------------------------------------------
# Goal
# ---------------------------------------------------------------------------

@dataclass
class Goal:
    """
    A goal opening on one end wall.

    scoring_team: the team that SCORES when the ball enters this goal.
      - left_goal  (x=0):            Team 1 scores (beat Team 0's defense)
      - right_goal (x=FIELD_DEPTH):  Team 0 scores (beat Team 1's defense)
    """
    x:            float
    scoring_team: int
    y_min:        float
    y_max:        float

    def contains(self, y: float) -> bool:
        """True if y is within the goal opening."""
        return self.y_min <= y <= self.y_max


# ---------------------------------------------------------------------------
# Ball state
# ---------------------------------------------------------------------------

@dataclass
class BallState:
    """
    The ball's position and velocity.

    x, y   — position on the field
    vx, vy — velocity (cm/s)
    """
    x:  float = 0.0
    y:  float = 0.0
    vx: float = 0.0
    vy: float = 0.0

    @property
    def speed(self) -> float:
        return math.sqrt(self.vx ** 2 + self.vy ** 2)

    @property
    def stopped(self) -> bool:
        return self.speed < config.STOP_THRESHOLD


# ---------------------------------------------------------------------------
# Team state
# ---------------------------------------------------------------------------

@dataclass
class TeamState:
    """
    Per-team mutable state for the continuous simulation.

    active_rods    : set of rod indices (into Field.rods) currently held.
    reaction_timer : seconds remaining before this team can change rod directions.
                     Set when the opponent hits the ball.
    """
    active_rods:    set[int] = dc_field(default_factory=set)
    reaction_timer: float    = 0.0

    @property
    def reacting(self) -> bool:
        """True if this team is still in reaction-time lockout."""
        return self.reaction_timer > 0

    def tick(self, dt: float) -> None:
        """Decrement the reaction timer."""
        if self.reaction_timer > 0:
            self.reaction_timer = max(0.0, self.reaction_timer - dt)


# ---------------------------------------------------------------------------
# Field
# ---------------------------------------------------------------------------

class Field:
    """
    The complete foosball table.

    Coordinate system
    -----------------
    x=0           : Team 0's goal line (left wall)
    x=FIELD_DEPTH : Team 1's goal line (right wall)
    y=0           : bottom side wall
    y=FIELD_WIDTH : top side wall

    Team 0 attacks rightward (+x direction).
    Team 1 attacks leftward  (-x direction).
    """

    def __init__(
        self,
        depth:           float        = config.FIELD_DEPTH,
        width:           float        = config.FIELD_WIDTH,
        goal_width:      float        = config.GOAL_WIDTH,
        rod_x_positions: list[float]  = None,
        rod_configs:     list[tuple]  = None,
        player_width:   float        = config.PLAYER_WIDTH,
        player_thickness:  float        = config.PLAYER_THICKNESS,
        rod_x_reach:     float        = config.ROD_X_REACH,
    ):
        self.depth = depth
        self.width = width

        # Goals (centered on y-axis)
        gy_min = (width - goal_width) / 2
        gy_max = (width + goal_width) / 2

        self.left_goal  = Goal(x=0.0,  scoring_team=1, y_min=gy_min, y_max=gy_max)
        self.right_goal = Goal(x=depth, scoring_team=0, y_min=gy_min, y_max=gy_max)

        # Build rods from config; (-1, -1) entries are blank slots (no rod)
        xs    = rod_x_positions or config.ROD_X_POSITIONS
        cfgs  = rod_configs     or config.ROD_CONFIGS
        self.rods: list[Rod] = [
            Rod(x=x, team=team, n_players=n,
                width=player_width, thickness=player_thickness,
                rod_x_reach=rod_x_reach)
            for x, (team, n) in zip(xs, cfgs)
            if team != -1
        ]

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def goal_for_attacker(self, team: int) -> Goal:
        """The goal the given team is attacking."""
        return self.right_goal if team == 0 else self.left_goal

    def rods_for_team(self, team: int) -> list[tuple[int, Rod]]:
        """Return [(index, rod), ...] for all rods belonging to `team`."""
        return [(i, r) for i, r in enumerate(self.rods) if r.team == team]

    def reset(self) -> None:
        """Reset every rod to defaults."""
        for rod in self.rods:
            rod.reset()

    def describe(self) -> str:
        """Human-readable field summary for debugging."""
        lines = [
            f"Field {self.depth}x{self.width} cm  |  "
            f"Goals: y=[{self.left_goal.y_min}, {self.left_goal.y_max}]",
            "",
            f"{'Idx':>3}  {'Team':>4}  {'x':>6}  {'Players':>7}  {'Positions (y)'}",
            "-" * 60,
        ]
        for i, rod in enumerate(self.rods):
            pos_str = ", ".join(f"{p:.1f}" for p in rod.player_positions)
            lines.append(
                f"{i:>3}  {rod.team:>4}  {rod.x:>6.1f}  {rod.n_players:>7}  [{pos_str}]"
            )
        return "\n".join(lines)
