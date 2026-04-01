"""
play.py — Run a single point and view/save the replay.

Usage:  python play.py
Tweak the settings below to change what you see.
"""

import os
import matplotlib
import matplotlib.pyplot as plt

# ---- Settings (edit these) ------------------------------------------------

SEED        = 3         # RNG seed (change for different games, None for random)
KICKOFF     = 1         # which team kicks off (0 or 1)
FPS         = 30        # ticks per second (higher = smoother but slower)
SHOW_REACH  = True      # draw player hitbox rectangles
SAVE_GIF    = True      # save to output/replay.gif
SHOW_LIVE   = False     # open a matplotlib window to watch live
TRACK_STATS = True      # show heatmap after the point

TEAM_0 = "HardOffense"
TEAM_1 = "TiltAndGap"

# Skill overrides (e.g. (0.9, 0.9) for (accuracy, power_consistency), None for default)
TEAM_0_SKILL = (0.7, 0.7)
TEAM_1_SKILL = (0.6, 0.7)

# ---------------------------------------------------------------------------

if not SHOW_LIVE:
    matplotlib.use("Agg")

import config
config.FPS = FPS
config.DT = 1.0 / FPS
config.MAX_TICKS = int(config.MAX_GAME_TIME * FPS)

from field import Field
from strategy import (SmackBall, AimAtGap, HardOffense,
                      DefensiveWall, TiltAndGap, ReactiveBlock)
import numpy as np
from simulation import simulate_point
from visualization import replay_point, draw_stats

STRATEGIES = {
    "SmackBall":     SmackBall,
    "AimAtGap":      AimAtGap,
    "HardOffense":   HardOffense,
    "DefensiveWall": DefensiveWall,
    "TiltAndGap":    TiltAndGap,
    "ReactiveBlock": ReactiveBlock,
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

    pos_grid  = np.zeros((int(field.depth / config.STATS_GRID_RES),
                           int(field.width  / config.STATS_GRID_RES))) if TRACK_STATS else None
    goal_hits: list = [] if TRACK_STATS else None

    result = simulate_point(
        field, strats,
        kickoff_team=KICKOFF,
        record=True,
        seed=SEED,
        pos_grid=pos_grid,
        goal_hits=goal_hits,
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
        title=f"{TEAM_0} vs {TEAM_1}",
    )

    if SHOW_LIVE:
        plt.show()
    elif save_path:
        os.startfile(os.path.abspath(save_path))

    # Heatmap
    if TRACK_STATS and pos_grid is not None:
        if pos_grid.max() > 0:
            pos_grid /= pos_grid.max()
        _, ax = plt.subplots(figsize=(14, 7), facecolor="#1a1a1a")
        draw_stats(field, pos_grid, goal_hits or [],
                   title=f"Heatmap — {TEAM_0} vs {TEAM_1}", ax=ax)
        if SHOW_LIVE:
            plt.show()
        else:
            heatmap_path = "output/heatmap.png"
            plt.savefig(heatmap_path, facecolor="#1a1a1a", dpi=120)
            print(f"Saved heatmap to {heatmap_path}")
            os.startfile(os.path.abspath(heatmap_path))


if __name__ == "__main__":
    main()
