"""
field.py — Core data structures for the foosball table.

Classes
-------
Rod        — A single foosball rod (fixed x, slides along y-axis).
Goal       — A goal opening on one end wall.
BallState  — The ball's current position and which rod possesses it.
TraceResult— The outcome of tracing a ball trajectory.
Field      — The complete table: rods, goals, and dimensions.
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
    - Fixed at x on the field.
    - Slides along the y-axis (all players move together).
    - Players are evenly spaced vertically; their default centers are at
      (i+1) * (FIELD_HEIGHT / (n_players + 1)) for i in 0..n_players-1.

    Skill parameters
    ----------------
    skill       : [0, 1] — controls shot angle std dev.
                  skill=1 → tight distribution (precise); skill=0 → wide (wild).
    consistency : [0, 1] — controls shot speed std dev.
                  consistency=1 → consistent pace; consistency=0 → erratic.

    Sliding
    -------
    y_offset shifts every player by the same amount.
    set_offset() clamps to keep all players inside the field boundaries.
    """

    def __init__(
        self,
        x: float,
        team: int,
        n_players: int,
        skill: float = 0.5,
        consistency: float = 0.5,
        y_reach: float = config.PLAYER_Y_REACH,
    ):
        self._base_x     = x
        self.team        = team
        self.n_players   = n_players
        self.skill       = skill
        self.consistency = consistency
        self.y_reach     = y_reach
        self.y_offset    = 0.0   # mutable sliding state (y-axis)
        self.x_offset    = 0.0   # mutable sliding state (x-axis)

        # x slide bounds: rod must stay within the field
        self._x_slide_min = -self._base_x
        self._x_slide_max = config.FIELD_WIDTH - self._base_x

        # Default player centers (equally spaced, centered on field height)
        spacing = config.FIELD_HEIGHT / (n_players + 1)
        self._base_positions: list[float] = [
            (i + 1) * spacing for i in range(n_players)
        ]

        # Valid offset range: outermost players must stay within [0, FIELD_HEIGHT]
        self._slide_min = -self._base_positions[0]  + y_reach
        self._slide_max =  config.FIELD_HEIGHT - self._base_positions[-1] - y_reach

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
        """(min_offset, max_offset) — valid sliding range."""
        return self._slide_min, self._slide_max

    @property
    def angle_std(self) -> float:
        """
        Shot angle std dev (radians) derived from skill.
        Linear interpolation: skill=0 → MAX_ANGLE_STD, skill=1 → MIN_ANGLE_STD.
        """
        return (
            config.MAX_ANGLE_STD
            + self.skill * (config.MIN_ANGLE_STD - config.MAX_ANGLE_STD)
        )

    @property
    def speed_std(self) -> float:
        """Speed std dev (cm/s) derived from consistency."""
        return (
            config.MAX_SPEED_STD
            + self.consistency * (config.MIN_SPEED_STD - config.MAX_SPEED_STD)
        )

    # ------------------------------------------------------------------
    # Methods
    # ------------------------------------------------------------------

    def set_offset(self, offset: float) -> None:
        """Slide rod to y `offset`, clamping to valid range."""
        self.y_offset = max(self._slide_min, min(self._slide_max, offset))

    def set_x_offset(self, offset: float) -> None:
        """Slide rod to x `offset`, clamping to keep the rod within the field."""
        self.x_offset = max(self._x_slide_min, min(self._x_slide_max, offset))

    def player_at_y(self, y: float) -> Optional[int]:
        """
        Return the index of the player whose hitbox covers y, or None.

        Used by physics.py during ray-casting: if the ball's trajectory
        passes through this rod's x-position at height y, and a player's
        hitbox covers y, possession transfers to this rod.
        """
        for i, py in enumerate(self.player_positions):
            if abs(y - py) <= self.y_reach:
                return i
        return None

    def reset(self) -> None:
        """Reset sliding position to default (centered)."""
        self.y_offset = 0.0
        self.x_offset = 0.0

    def __repr__(self) -> str:
        return (
            f"Rod(x={self.x}, team={self.team}, n={self.n_players}, "
            f"skill={self.skill:.2f}, offset={self.y_offset:.1f})"
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
      - right_goal (x=FIELD_WIDTH):  Team 0 scores (beat Team 1's defense)
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
    The ball's current state when a rod has possession.

    x, y               — position on the field (x = rod.x, y = player center)
    possessing_rod_idx — index into Field.rods
    """
    x:                  float
    y:                  float
    possessing_rod_idx: int


# ---------------------------------------------------------------------------
# Trace result
# ---------------------------------------------------------------------------

@dataclass
class TraceResult:
    """
    What happened at the end of a ball trajectory trace.

    kind: 'goal'       — ball entered a goal; `scoring_team` is set.
          'possession'  — ball intercepted by a player; `rod_idx`, `x`, `y` are set.
          'dead'        — ball left the field or max bounces exceeded; point resets.
    """
    kind:         str
    scoring_team: int   = -1
    rod_idx:      int   = -1
    x:            float = 0.0
    y:            float = 0.0


# ---------------------------------------------------------------------------
# Field
# ---------------------------------------------------------------------------

class Field:
    """
    The complete foosball table.

    Owns the list of Rod objects and the two Goal objects.
    All rod y-offsets are mutable (strategies slide rods each turn).

    Coordinate system
    -----------------
    x=0           : Team 0's goal line (left wall)
    x=FIELD_WIDTH : Team 1's goal line (right wall)
    y=0           : bottom side wall
    y=FIELD_HEIGHT: top side wall

    Team 0 attacks rightward (+x direction).
    Team 1 attacks leftward  (-x direction).
    """

    def __init__(
        self,
        width:           float        = config.FIELD_WIDTH,
        height:          float        = config.FIELD_HEIGHT,
        goal_width:      float        = config.GOAL_WIDTH,
        rod_x_positions: list[float]  = None,
        rod_configs:     list[tuple]  = None,
        player_y_reach:  float        = config.PLAYER_Y_REACH,
    ):
        self.width  = width
        self.height = height

        # Goals (centered on y-axis)
        gy_min = (height - goal_width) / 2
        gy_max = (height + goal_width) / 2

        # Ball entering the LEFT  goal → Team 1 scored (broke through Team 0's defense)
        # Ball entering the RIGHT goal → Team 0 scored (broke through Team 1's defense)
        self.left_goal  = Goal(x=0.0,  scoring_team=1, y_min=gy_min, y_max=gy_max)
        self.right_goal = Goal(x=width, scoring_team=0, y_min=gy_min, y_max=gy_max)

        # Build rods from config
        xs    = rod_x_positions or config.ROD_X_POSITIONS
        cfgs  = rod_configs     or config.ROD_CONFIGS
        self.rods: list[Rod] = [
            Rod(x=x, team=team, n_players=n, y_reach=player_y_reach)
            for x, (team, n) in zip(xs, cfgs)
        ]

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def goal_for_attacker(self, team: int) -> Goal:
        """The goal the given team is attacking (i.e., trying to score in)."""
        # Team 0 attacks right → right_goal; Team 1 attacks left → left_goal
        return self.right_goal if team == 0 else self.left_goal

    def rods_for_team(self, team: int) -> list[tuple[int, Rod]]:
        """Return [(index, rod), ...] for all rods belonging to `team`."""
        return [(i, r) for i, r in enumerate(self.rods) if r.team == team]

    def reset_rod_offsets(self) -> None:
        """Reset every rod to its default (centered) sliding position."""
        for rod in self.rods:
            rod.reset()

    def describe(self) -> str:
        """Human-readable field summary for debugging."""
        lines = [
            f"Field {self.width}x{self.height} cm  |  "
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
