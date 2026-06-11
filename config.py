"""
config.py — Global constants for the foosball simulation.

All physical dimensions are in centimeters (cm).
All speeds are in cm/s.
All times are in seconds.
All angles are in radians.
Accuracy/power_consistency values are in [0, 1].
"""

import math

# ---------------------------------------------------------------------------
# Field dimensions
# ---------------------------------------------------------------------------
FIELD_DEPTH  = 120.0   # x-axis: Team 0 goal (x=0) → Team 1 goal (x=FIELD_DEPTH)
FIELD_WIDTH  =  68.0   # y-axis: bottom wall (y=0) → top wall (y=FIELD_WIDTH)

# Goal opening, centered on the y-axis
GOAL_WIDTH = 20.0
GOAL_DEPTH = 3.0   # cm — ball must travel this far past the goal line to score
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
ROD_X_POSITIONS = [9.0, 23.6, 38.1, 52.7, 67.3, 81.9, 96.4, 111.0]

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
PLAYER_WIDTH     = 2.5   # cm — width of a player figure (y direction)
PLAYER_THICKNESS = 1.5   # cm — thickness of a player figure (x direction)

# Rod rotation modelled as x-slide: max distance the rod center can move
# from its default x origin (player figure sweep distance)
ROD_X_REACH = 3.0      # cm — max x-offset from rod origin

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
BALL_RADIUS    = 1.75     # cm — ball radius (~35 mm diameter, real foosball ball)
FRICTION       = 10.0     # cm/s² — constant deceleration (ball slows each tick)
STOP_THRESHOLD = 3.0      # cm/s — below this, ball counts as stopped
BALL_MAX_SPEED = 260.0    # cm/s — absolute cap on ball speed
HIT_SPEED      = 140.0    # cm/s — default intended shot speed

# ---------------------------------------------------------------------------
# Shot noise — maps accuracy/power_consistency (0-1) to distribution std devs
#
#   accuracy=0  → MAX_ANGLE_STD  (wild, unpredictable shots)
#   accuracy=1  → MIN_ANGLE_STD  (tight, precise shots)
#
#   power_consistency=0  → MAX_SPEED_STD  (very variable speed)
#   power_consistency=1  → MIN_SPEED_STD  (very consistent speed)
# ---------------------------------------------------------------------------
MIN_ANGLE_STD = math.radians(4)
MAX_ANGLE_STD = math.radians(25)

MIN_SPEED_STD =  3.0   # cm/s
MAX_SPEED_STD = 40.0   # cm/s

MIN_MOVEMENT_STD = 0.1*PLAYER_WIDTH   # cm — at movement_control=1 (precise positioning)
MAX_MOVEMENT_STD = 1.1*PLAYER_WIDTH   # cm — at movement_control=0 (sloppy positioning)

# ---------------------------------------------------------------------------
# Passive player contact — ball hitting an uncontrolled rod's player
# ---------------------------------------------------------------------------
CONTACT_SLOWDOWN    = 0.8    # vx multiplier on contact (0 = full stop, 1 = no effect)
CONTROLLED_SLOWDOWN = 0.6    # speed multiplier on idle controlled-rod bounce / whiff
CONTACT_MIN_SPEED   = 5.0   # cm/s — below this, vx zeroes out on contact
SPEED_PUSHBACK      = 0.005  # x_offset pushback per unit of ball speed
OFFSET_PUSHBACK     = 0.10   # x_offset pushback per unit of offset distance from pushback_x_start
PUSHBACK_X_START    = 2.4    # cm — rod this far forward will not be pushed back
PASSTHROUGH_SPEED   = BALL_MAX_SPEED * 0.9  # cm/s — ball faster than this flips the rod up
ROD_UP_THRESHOLD    = -2.4   # cm — rod pushed back past this x_offset flips up
GLANCE_THRESHOLD    = -1.5   # cm — rod must be tilted back this far for a glance
CONTACT_DEFLECTION  = 0.30   # vy deflection factor — fraction of vx added to vy based on hit position

# ---------------------------------------------------------------------------
# Rod movement
# ---------------------------------------------------------------------------
MOVEMENT_SPEED = 150.0     # cm/s — max rod slide speed toward target_y

# ---------------------------------------------------------------------------
# Timing parameters (defaults — can be overridden per team)
# ---------------------------------------------------------------------------
REACTION_TIME  = 0.30     # seconds — delay before opponent can change direction after a hit
SWITCH_DELAY   = 0.50     # seconds — delay when switching a hand to a new rod

# ---------------------------------------------------------------------------
# Swing commitment — proactive hit timing (see swing-window plan)
#
# A controlled rod commits a swing in advance. The swing has a fixed physical
# shape: a backswing, then a short active window during which contact connects.
# Scaled to whole ticks (DT) so the window spans observable ticks at 10 FPS.
#
#   commit at t_c →
#     active_start = t_c + SWING_DURATION * BACKSWING_RATIO
#     window_end   = t_c + SWING_DURATION
#   Ball must reach the player within [active_start, window_end] to be struck;
#   otherwise the held rod rigid-bounces (a "whiff").
# ---------------------------------------------------------------------------
SWING_DURATION  = 0.4    # seconds — total swing (backswing + active window)
BACKSWING_RATIO = 0.5    # fraction of the swing spent in backswing
COMMIT_COOLDOWN = SWING_DURATION  # seconds — min time between successive swing commits

# Anticipation skill — accuracy of the time-to-contact (TTC) estimate.
#   anticipation=1 → near-perfect TTC even at long horizon (commit early, far away)
#   anticipation=0 → TTC noise grows fast with horizon (reliable only up close)
#   sigma(ttc) = ANTICIPATION_TTC_NOISE * ttc * (1 - anticipation)
DEFAULT_ANTICIPATION    = 0.7
ANTICIPATION_TTC_NOISE  = 0.5    # base TTC noise per second of prediction horizon
ANTICIPATION_MAX_HORIZON = 1.0   # seconds — don't attempt to commit beyond this TTC

# ---------------------------------------------------------------------------
# Gameplay limits
# ---------------------------------------------------------------------------
N_HANDS = 2   # max rods a team can control simultaneously (1 to n_team_rods)
POSSESSION_LIMIT = 8.0  # seconds — one team continuously in reach, opponent not → loses point

# ---------------------------------------------------------------------------
# Stats / heatmap
# ---------------------------------------------------------------------------
STATS_GRID_RES = 1.0   # cm per grid cell (lower = finer resolution)
