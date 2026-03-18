"""
config.py — Global constants for the foosball simulation.

All physical dimensions are in centimeters (cm).
All speeds are in cm/s.
All times are in seconds.
All angles are in radians.
Skill/consistency values are in [0, 1].
"""

import math

# ---------------------------------------------------------------------------
# Field dimensions
# ---------------------------------------------------------------------------
FIELD_DEPTH  = 120.0   # x-axis: Team 0 goal (x=0) → Team 1 goal (x=FIELD_DEPTH)
FIELD_WIDTH  =  68.0   # y-axis: bottom wall (y=0) → top wall (y=FIELD_WIDTH)

# Goal opening, centered on the y-axis
GOAL_WIDTH = 20.0
GOAL_Y_MIN = (FIELD_WIDTH - GOAL_WIDTH) / 2   # 24.0 cm
GOAL_Y_MAX = (FIELD_WIDTH + GOAL_WIDTH) / 2   # 44.0 cm

# ---------------------------------------------------------------------------
# Rod layout — 8 rods, left-to-right, alternating teams
#
# Team 0 attacks rightward (+x) and defends the left goal  (x = 0)
# Team 1 attacks leftward  (-x) and defends the right goal (x = FIELD_DEPTH)
#
# Each entry: (team_id, n_players)  |  (-1, -1) = blank slot (no rod)
# Using the 3-goalie variant: goalie rods have 3 players instead of 1.
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
PLAYER_WIDTH     = 2.5   # cm — full width of a player figure (y direction)
PLAYER_THICKNESS = 1.4   # cm — full thickness of a player figure (x direction)

# Rod rotation modelled as x-slide: max distance the rod center can move
# from its default x origin (player figure sweep distance)
ROD_X_REACH = 2.5      # cm — max x-offset from rod origin

# ---------------------------------------------------------------------------
# Time-stepped simulation
# ---------------------------------------------------------------------------
FPS            = 10       # ticks per second (increase for accuracy, decrease for speed)
DT             = 1.0 / FPS
MAX_GAME_TIME  = 60.0     # seconds — point ends if exceeded
MAX_TICKS      = int(MAX_GAME_TIME * FPS)

# ---------------------------------------------------------------------------
# Ball physics
# ---------------------------------------------------------------------------
FRICTION       = 15.0     # cm/s² — constant deceleration (ball slows each tick)
STOP_THRESHOLD =  2.0     # cm/s — below this, ball counts as stopped
BALL_MAX_SPEED = 200.0    # cm/s — absolute cap on ball speed

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

# ---------------------------------------------------------------------------
# Rod movement
# ---------------------------------------------------------------------------
MOVEMENT_SPEED = 80.0     # cm/s — max rod slide speed toward target_y

# ---------------------------------------------------------------------------
# Timing / skill parameters (defaults — can be overridden per team)
# ---------------------------------------------------------------------------
REACTION_TIME  = 0.15     # seconds — delay before opponent can change direction after a hit
SWITCH_DELAY   = 0.10     # seconds — delay when switching a hand to a new rod

# ---------------------------------------------------------------------------
# Gameplay limits
# ---------------------------------------------------------------------------
N_HANDS = 2   # max rods a team can control simultaneously (1 to n_team_rods)
