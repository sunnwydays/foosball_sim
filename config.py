"""
config.py — Global constants for the foosball simulation.

All physical dimensions are in centimeters (cm).
All angles are in radians.
Skill/consistency values are in [0, 1].
"""

import math

# ---------------------------------------------------------------------------
# Field dimensions
# ---------------------------------------------------------------------------
FIELD_WIDTH  = 120.0   # x-axis: Team 0 goal (x=0) → Team 1 goal (x=FIELD_WIDTH)
FIELD_HEIGHT =  68.0   # y-axis: bottom wall (y=0) → top wall (y=FIELD_HEIGHT)

# Goal opening, centered on the y-axis
GOAL_WIDTH = 20.0
GOAL_Y_MIN = (FIELD_HEIGHT - GOAL_WIDTH) / 2   # 24.0 cm
GOAL_Y_MAX = (FIELD_HEIGHT + GOAL_WIDTH) / 2   # 44.0 cm

# ---------------------------------------------------------------------------
# Rod layout — 8 rods, left-to-right, alternating teams
#
# Team 0 attacks rightward (+x) and defends the left goal  (x = 0)
# Team 1 attacks leftward  (-x) and defends the right goal (x = FIELD_WIDTH)
#
# Each entry: (team_id, n_players)
# Using the 3-goalie variant: goalie rods have 3 players instead of 1,
# so they can slide to cover corners and keep the model purely 2D.
# ---------------------------------------------------------------------------
ROD_X_POSITIONS = [9.0, 24.0, 38.0, 54.0, 66.0, 82.0, 96.0, 111.0]

ROD_CONFIGS = [
    (0, 3),   # index 0 — Team 0 Goalie    (3-player variant)
    (0, 2),   # index 1 — Team 0 Defense
    (1, 3),   # index 2 — Team 1 Forward
    (0, 5),   # index 3 — Team 0 Midfield   <- Team 0 kickoff rod
    (1, 5),   # index 4 — Team 1 Midfield   <- Team 1 kickoff rod
    (0, 3),   # index 5 — Team 0 Forward
    (1, 2),   # index 6 — Team 1 Defense
    (1, 3),   # index 7 — Team 1 Goalie    (3-player variant)
]

# Which rod index kicks off for each team (their midfield rod)
KICKOFF_ROD = {0: 3, 1: 4}

# ---------------------------------------------------------------------------
# Player physical parameters
# ---------------------------------------------------------------------------
PLAYER_Y_REACH = 3.5   # cm — half-width of a player's interception hitbox in y

# ---------------------------------------------------------------------------
# Shot noise — maps skill/consistency (0–1) to distribution std devs
#
#   skill=0  → MAX_ANGLE_STD  (wild, unpredictable shots)
#   skill=1  → MIN_ANGLE_STD  (tight, precise shots)
#
#   consistency=0  → MAX_SPEED_STD  (very variable speed)
#   consistency=1  → MIN_SPEED_STD  (very consistent speed)
# ---------------------------------------------------------------------------
MIN_ANGLE_STD = math.radians(3)    #  ~3° for near-perfect skill
MAX_ANGLE_STD = math.radians(35)   # ~35° for low skill

MIN_SPEED_STD =  2.0   # cm/s
MAX_SPEED_STD = 25.0   # cm/s

# Absolute speed bounds
MIN_SPEED =  10.0   # cm/s  (floor — a shot always has some force)
MAX_SPEED = 100.0   # cm/s  (ceiling)

# ---------------------------------------------------------------------------
# Simulation limits — prevent infinite loops
# ---------------------------------------------------------------------------
MAX_WALL_BOUNCES = 50   # max wall reflections per single shot
MAX_TURNS        = 100  # max possession changes per point
