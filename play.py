"""
play.py — Run a single point and view/save the replay.

Usage:  python play.py [seed]
Tweak the settings below to change what you see.
Pass a seed as an argument to override SEED for this run.
"""

import os
import sys
from datetime import datetime
import matplotlib

# ---- Settings (edit these) ------------------------------------------------

SEED        = 36                # RNG seed (None for random)
TEAM_0      = "HardOffense"     # strategy name or path to a genome .json
TEAM_1      = "TiltAndGap"      # e.g. "output/agents/rank0.json"

KICKOFF     = 1         # kickoff team (0 or 1)
FPS         = 30        # ticks per second (higher = smoother but slower)

GIF_SLOWDOWN = 1        # >1 saves GIF at lower fps (e.g. 2 = half-speed)
ANIMATE     = False     # build the replay animation and save output/replay.gif (slow)
INTERACTIVE = True      # open a scrubbable viewer window instead of the GIF (overrides ANIMATE)
SHOW_REACH  = True      # draw player hitbox rectangles
SAVE_HEATMAP = True     # show heatmap after the point
SAVE_ACTION_LOG = True  # print action table to terminal + save JSON

# Skill overrides (e.g. (0.9, 0.9) for (accuracy, power_consistency), None for default)
# Ignored for genome-based agents (skill genes are in the genome itself)
TEAM_0_SKILL = (0.7, 0.7)
TEAM_1_SKILL = (0.6, 0.7)

# Anticipation overrides (0.0–1.0, or None to use config.DEFAULT_ANTICIPATION)
# Ignored for genome-based agents (anticipation gene is in the genome itself)
TEAM_0_ANTICIPATION = None
TEAM_1_ANTICIPATION = None

# Local overrides: copy above settings and change in play_local.py (gitignored)
try:
    from play_local import *
except ImportError:
    pass

# The interactive viewer needs a GUI backend; only force the headless Agg
# backend when we are just writing files (GIF/heatmap) so the GIF render is fast.
if not INTERACTIVE:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------

import config
config.FPS = FPS
config.DT = 1.0 / FPS
config.MAX_TICKS = int(config.MAX_GAME_TIME * FPS)

from field import Field
from strategy import (SmackBall, AimAtGap, HardOffense,
                      DefensiveWall, TiltAndGap, ReactiveBlock)
import numpy as np
from simulation import simulate_point
from visualization import replay_point, draw_stats, interactive_replay

STRATEGIES = {
    "SmackBall":     SmackBall,
    "AimAtGap":      AimAtGap,
    "HardOffense":   HardOffense,
    "DefensiveWall": DefensiveWall,
    "TiltAndGap":    TiltAndGap,
    "ReactiveBlock": ReactiveBlock,
}

def build_strategy(spec: str):
    if spec.endswith(".json"):
        from evolution import load_agent, ParameterizedStrategy
        agent, meta = load_agent(spec)
        print(f"Loaded genome from {spec}" + (f"  wr={meta['win_rate']}" if "win_rate" in meta else ""))
        return ParameterizedStrategy(agent.genome)
    return STRATEGIES[spec]()


_PHYSICS_ACTIONS = {"DEFLECT", "STOP"}

def _is_physics_action(action: str) -> bool:
    return action in _PHYSICS_ACTIONS or action.startswith("WALL_")

def _format_event(e) -> str:
    if e.rod_label.startswith("GOAL"):
        return f"  t={e.game_time:5.2f}s  *** {e.action} ***"
    if _is_physics_action(e.action):
        before = f"(vx={e.intended_vel[0]:+.2f},vy={e.intended_vel[1]:+.2f})"
        after  = f"(vx={e.actual_vel[0]:+.2f},vy={e.actual_vel[1]:+.2f})"
        return (f"  t={e.game_time:5.2f}s  {e.action:<9}  "
                f"ball=({e.ball_pos[0]:.1f},{e.ball_pos[1]:.1f})  "
                f"{before} -> {after}  reach=[{e.rod_label}]")
    if e.action == "COMMIT":
        vel = f"  intent=(vx={e.intended_vel[0]:+.2f}, vy={e.intended_vel[1]:+.2f})"
    elif e.action == "HIT":
        vel = f"  actual=(vx={e.actual_vel[0]:+.2f}, vy={e.actual_vel[1]:+.2f})"
    elif e.action == "WHIFF":
        vel = f"  intent=(vx={e.intended_vel[0]:+.2f}, vy={e.intended_vel[1]:+.2f})"
    else:
        vel = ""
    return (f"  t={e.game_time:5.2f}s  {e.rod_label:<10}  {e.action:<7}  "
            f"ball=({e.ball_pos[0]:.1f}, {e.ball_pos[1]:.1f}){vel}")


def _dump_action_log(log, label0: str, label1: str, seed: int) -> None:
    os.makedirs("output", exist_ok=True)
    counts: dict[str, int] = {}
    for e in log:
        if not e.rod_label.startswith("GOAL"):
            counts[e.action] = counts.get(e.action, 0) + 1
    summary = "  ".join(f"{k}:{v}" for k, v in counts.items() if v)

    lines = [
        f"team_0: {label0}  team_1: {label1}  seed: {seed}",
        f"\n=== Action Log ({len(log)} events - {summary}) ===",
        *(_format_event(e) for e in log),
    ]
    text = "\n".join(lines)

    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"output/action_log_{ts}.txt"
    with open(path, "w") as f:
        f.write(text + "\n")

    print(text)
    print(f"Saved action log to {path}")


def main():
    seed = SEED
    if len(sys.argv) > 1:
        seed = None if sys.argv[1].lower() in ("none", "random") else int(sys.argv[1])
    if seed is None:
        seed = int(np.random.randint(0, 2**31))

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
    if TEAM_0_ANTICIPATION is not None:
        for rod in field.rods:
            if rod.team == 0:
                rod.anticipation = TEAM_0_ANTICIPATION
    if TEAM_1_ANTICIPATION is not None:
        for rod in field.rods:
            if rod.team == 1:
                rod.anticipation = TEAM_1_ANTICIPATION

    strats = {
        0: build_strategy(TEAM_0),
        1: build_strategy(TEAM_1),
    }

    pos_grid  = np.zeros((int(field.depth / config.STATS_GRID_RES),
                           int(field.width  / config.STATS_GRID_RES))) if SAVE_HEATMAP else None
    goal_hits: list = [] if SAVE_HEATMAP else None

    result = simulate_point(
        field, strats,
        kickoff_team=KICKOFF,
        record=True,
        seed=seed,
        pos_grid=pos_grid,
        goal_hits=goal_hits,
        collect_action_log=SAVE_ACTION_LOG,
    )

    winner_str = f"Team {result.winner}" if result.winner is not None else "Draw"
    hits = sum(1 for f in result.frames if f.event == "hit")
    print(f"Result: {winner_str}  |  {result.ticks} ticks  |  {result.time:.2f}s  |  {hits} hits")

    # Replay. The interactive viewer supersedes the GIF render when both are on.
    save_path = "output/replay.gif" if (ANIMATE and not INTERACTIVE) else None
    label0 = os.path.basename(TEAM_0).replace(".json", "") if TEAM_0.endswith(".json") else TEAM_0
    label1 = os.path.basename(TEAM_1).replace(".json", "") if TEAM_1.endswith(".json") else TEAM_1

    if SAVE_ACTION_LOG and result.action_log:
        _dump_action_log(result.action_log, label0, label1, seed)

    # Heatmap (fast — open first)
    if SAVE_HEATMAP and pos_grid is not None:
        if pos_grid.max() > 0:
            pos_grid /= pos_grid.max()
        _, ax = plt.subplots(figsize=(10, 6), facecolor="#1a1a1a")
        fig = ax.get_figure()
        fig.subplots_adjust(left=0.07, right=0.93, top=0.94, bottom=0.07)
        draw_stats(field, pos_grid, goal_hits or [],
                   title=f"Heatmap — {label0} vs {label1}", ax=ax)
        heatmap_path = "output/heatmap.png"
        plt.savefig(heatmap_path, facecolor="#1a1a1a", dpi=100)
        print(f"Saved heatmap to {heatmap_path}")
        os.startfile(os.path.abspath(heatmap_path))
        plt.close(fig)   # keep it out of the interactive viewer's plt.show()

    if ANIMATE and not INTERACTIVE:
        anim = replay_point(
            field, result.frames,
            show_reach=SHOW_REACH,
            save_path=save_path,
            interval=max(1, 1000 // FPS),
            gif_slowdown=GIF_SLOWDOWN,
            title=f"{label0} vs {label1}",
        )

        if save_path:
            os.startfile(os.path.abspath(save_path))

    if INTERACTIVE:
        interactive_replay(
            field, result.frames,
            show_reach=SHOW_REACH,
            fps=FPS,
            title=f"{label0} vs {label1}",
        )


if __name__ == "__main__":
    main()
