"""
observation.py -- what the RL agent sees each tick.

build_obs(view, team) turns the live sim state (a simulation.TickView) into a
fixed-length float vector in roughly [-1, 1].

Mirroring
---------
Team 1 attacks toward -x, team 0 toward +x. The rod x positions are symmetric
about the field centre (9 <-> 111, 23.6 <-> 96.4, ...), so for team 1 we reflect
the world: x -> FIELD_DEPTH - x, vx -> -vx, x_offset -> -x_offset. After that
the agent always defends x=0 and attacks +x, and one network plays both sides.
Actions need no mirroring: the aim helpers are already team-aware.

Layout (OBS_DIM floats)
-----------------------
  ball     : x, y, vx, vy                                   (4)
  8 rods   : own 4 then opponent 4, each ordered from that team's own goal
             outward (goalie, defense, mid, forward). Per rod:
             y_offset, x_offset, up, controlled, swing armed   (5 each = 40)
  context  : own reacting flag, own / opponent possession timer   (3)
"""

import numpy as np

import config
from simulation import TickView

N_ROD_FEATURES = 5
OBS_DIM = 4 + 8 * N_ROD_FEATURES + 3


def build_obs(view: TickView, team: int) -> np.ndarray:
    ball, field, team_states, possession_timer, _ = view
    sign = 1.0 if team == 0 else -1.0   # +1: no mirror, -1: mirror x
    depth, width = field.depth, field.width

    obs = np.empty(OBS_DIM, dtype=np.float32)
    bx = ball.x if team == 0 else depth - ball.x
    obs[0] = 2.0 * bx / depth - 1.0
    obs[1] = 2.0 * ball.y / width - 1.0
    obs[2] = sign * ball.vx / config.BALL_MAX_SPEED
    obs[3] = ball.vy / config.BALL_MAX_SPEED

    # Own rods ascend in base x for team 0 and descend for team 1, so index 0
    # is always the goalie. The opponent's rods run the other way for the same reason.
    by_x = sorted(field.rods, key=lambda r: r._base_x)
    own = [r for r in by_x if r.team == team]
    opp = [r for r in by_x if r.team != team]
    if team == 1:
        own.reverse()
    else:
        opp.reverse()

    i = 4
    for rod in own + opp:
        obs[i]     = rod.y_offset / rod._slide_max
        obs[i + 1] = sign * rod.x_offset / config.ROD_X_REACH
        obs[i + 2] = float(rod.up)
        obs[i + 3] = float(rod.controlled)
        obs[i + 4] = float(rod.pending_swing is not None)
        i += N_ROD_FEATURES

    obs[i]     = float(team_states[team].reacting)
    obs[i + 1] = possession_timer[team] / config.POSSESSION_LIMIT
    obs[i + 2] = possession_timer[1 - team] / config.POSSESSION_LIMIT
    return np.clip(obs, -1.0, 1.0, out=obs)
