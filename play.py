"""
play.py — Run a single point and view/save the replay.

Usage:  python play.py [seed]
Tweak the settings below to change what you see.
Pass a seed as an argument to override SEED for this run.
"""

import json
import os
import sys
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- Settings (edit these) ------------------------------------------------

SEED        = 36        # RNG seed (change for different games, None for random)
KICKOFF     = 1         # which team kicks off (0 or 1)
FPS         = 30        # ticks per second (higher = smoother but slower)
SHOW_REACH  = True      # draw player hitbox rectangles
ANIMATE     = True      # build the replay animation and save output/replay.gif (slow)
TRACK_STATS = True      # show heatmap after the point

TEAM_0 = "HardOffense"          # strategy name or path to a genome .json
TEAM_1 = "TiltAndGap"           # e.g. "output/agents/rank0.json"

SAVE_ACTION_LOG = True   # print action table to terminal + save JSON
EXTENDED_LOG    = False  # also log every ball direction change

# Skill overrides (e.g. (0.9, 0.9) for (accuracy, power_consistency), None for default)
# Ignored for genome-based agents (skill genes are in the genome itself)
TEAM_0_SKILL = (0.7, 0.7)
TEAM_1_SKILL = (0.6, 0.7)

# Anticipation overrides (0.0–1.0, or None to use config.DEFAULT_ANTICIPATION)
# Ignored for genome-based agents (anticipation gene is in the genome itself)
TEAM_0_ANTICIPATION = None
TEAM_1_ANTICIPATION = None

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
from visualization import replay_point, draw_stats

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


_PHYSICS_ACTIONS = {"BOUNCE_X", "BOUNCE_Y", "DEFLECT", "STOP"}

def _print_action_log(log) -> None:
    counts: dict[str, int] = {}
    for e in log:
        counts[e.action] = counts.get(e.action, 0) + 1
    summary = "  ".join(f"{k}:{v}" for k, v in counts.items() if v)
    print(f"\n=== Action Log ({len(log)} events - {summary}) ===")
    for e in log:
        if e.action in _PHYSICS_ACTIONS:
            before = f"(vx={e.intended_vel[0]:+.2f},vy={e.intended_vel[1]:+.2f})"
            after  = f"(vx={e.actual_vel[0]:+.2f},vy={e.actual_vel[1]:+.2f})"
            print(f"  t={e.game_time:5.2f}s  {e.action:<9}  "
                  f"ball=({e.ball_pos[0]:.1f},{e.ball_pos[1]:.1f})  "
                  f"{before} -> {after}  reach=[{e.rod_label}]")
        else:
            if e.action == "COMMIT":
                vel = f"  intent=(vx={e.intended_vel[0]:+.2f}, vy={e.intended_vel[1]:+.2f})"
            elif e.action == "HIT":
                vel = f"  actual=(vx={e.actual_vel[0]:+.2f}, vy={e.actual_vel[1]:+.2f})"
            elif e.action == "WHIFF":
                vel = f"  intent=(vx={e.intended_vel[0]:+.2f}, vy={e.intended_vel[1]:+.2f})"
            else:
                vel = ""
            print(f"  t={e.game_time:5.2f}s  {e.rod_label:<10}  {e.action:<7}  "
                  f"ball=({e.ball_pos[0]:.1f}, {e.ball_pos[1]:.1f}){vel}")


def _save_action_log(log, label0: str, label1: str) -> None:
    os.makedirs("output", exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"output/action_log_{ts}.json"
    data = {
        "team_0": label0,
        "team_1": label1,
        "events": [
            {
                "time":         e.game_time,
                "team":         e.team,
                "rod":          e.rod_label,
                "action":       e.action,
                "ball_pos":     list(e.ball_pos),
                "intended_vel": list(e.intended_vel) if e.intended_vel else None,
                "actual_vel":   list(e.actual_vel)   if e.actual_vel   else None,
            }
            for e in log
        ],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved action log to {path}")


def main():
    seed = SEED
    if len(sys.argv) > 1:
        seed = None if sys.argv[1].lower() in ("none", "random") else int(sys.argv[1])

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
                           int(field.width  / config.STATS_GRID_RES))) if TRACK_STATS else None
    goal_hits: list = [] if TRACK_STATS else None

    result = simulate_point(
        field, strats,
        kickoff_team=KICKOFF,
        record=True,
        seed=seed,
        pos_grid=pos_grid,
        goal_hits=goal_hits,
        collect_action_log=SAVE_ACTION_LOG,
        extended_log=EXTENDED_LOG,
    )

    winner_str = f"Team {result.winner}" if result.winner is not None else "Draw"
    hits = sum(1 for f in result.frames if f.event == "hit")
    print(f"Result: {winner_str}  |  {result.ticks} ticks  |  {result.time:.2f}s  |  {hits} hits")

    # Replay
    save_path = "output/replay.gif" if ANIMATE else None
    label0 = os.path.basename(TEAM_0).replace(".json", "") if TEAM_0.endswith(".json") else TEAM_0
    label1 = os.path.basename(TEAM_1).replace(".json", "") if TEAM_1.endswith(".json") else TEAM_1

    if SAVE_ACTION_LOG and result.action_log:
        _print_action_log(result.action_log)
        _save_action_log(result.action_log, label0, label1)

    # Heatmap (fast — open first)
    if TRACK_STATS and pos_grid is not None:
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

    if ANIMATE:
        anim = replay_point(
            field, result.frames,
            show_reach=SHOW_REACH,
            save_path=save_path,
            interval=max(1, 1000 // FPS),
            title=f"{label0} vs {label1}",
        )

        if save_path:
            os.startfile(os.path.abspath(save_path))


if __name__ == "__main__":
    main()
