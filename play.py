"""
play.py — Run a single point and view/save the replay.

Usage:  python play.py
Tweak the settings below to change what you see.
"""

import os
import matplotlib
import matplotlib.pyplot as plt

# ---- Settings (edit these) ------------------------------------------------

SEED        = 4         # RNG seed (change for different games, None for random)
KICKOFF     = 1         # which team kicks off (0 or 1)
FPS         = 30        # ticks per second (higher = smoother but slower)
SHOW_REACH  = True      # draw player hitbox rectangles
SAVE_GIF    = True      # save to output/replay.gif
SHOW_LIVE   = True     # open a matplotlib window to watch live

# Team 0 strategy
TEAM_0 = "SmackBall"    # "SmackBall" or "AimAtGap"
# Team 1 strategy
TEAM_1 = "AimAtGap"

# Skill overrides (None = use defaults)
TEAM_0_SKILL = None     # e.g. (0.9, 0.9) for (accuracy, power_consistency)
TEAM_1_SKILL = None     # e.g. (0.2, 0.2)

# ---------------------------------------------------------------------------

if not SHOW_LIVE:
    matplotlib.use("Agg")

import config
config.FPS = FPS
config.DT = 1.0 / FPS
config.MAX_TICKS = int(config.MAX_GAME_TIME * FPS)

from field import Field
from strategy import SmackBall, AimAtGap
from simulation import simulate_point
from visualization import replay_point

STRATEGIES = {
    "SmackBall": SmackBall,
    "AimAtGap":  AimAtGap,
}


def main():
    field = Field()

    # Apply skill overrides
    if TEAM_0_SKILL:
        for rod in field.rods:
            if rod.team == 0:
                rod.accuracy, rod.power_consistency = TEAM_0_SKILL
    if TEAM_1_SKILL:
        for rod in field.rods:
            if rod.team == 1:
                rod.accuracy, rod.power_consistency = TEAM_1_SKILL

    strats = {
        0: STRATEGIES[TEAM_0](),
        1: STRATEGIES[TEAM_1](),
    }

    result = simulate_point(
        field, strats,
        kickoff_team=KICKOFF,
        record=True,
        seed=SEED,
    )

    winner_str = f"Team {result.winner}" if result.winner is not None else "Draw"
    hits = sum(1 for f in result.frames if f.event == "hit")
    print(f"Result: {winner_str}  |  {result.ticks} ticks  |  {result.time:.2f}s  |  {hits} hits")

    # Replay
    save_path = "output/replay.gif" if SAVE_GIF else None
    anim = replay_point(
        field, result.frames,
        show_reach=SHOW_REACH,
        save_path=save_path,
        interval=max(1, 1000 // FPS),
    )

    if SHOW_LIVE:
        plt.show()
    elif save_path:
        os.startfile(os.path.abspath(save_path))


if __name__ == "__main__":
    main()
