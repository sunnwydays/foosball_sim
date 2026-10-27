"""golden_points.py -- exact-output regression check for simulate_point.

Records, for seeds 0..199 on two matchups, everything observable about a point:
(winner, ticks, outcome, is_self_goal, hash of the recorded frames and the
action log). Run --record on the frozen commit BEFORE refactoring
simulate_point, then --check afterwards: any change in RNG call order or tick
logic shows up as a mismatch.

Matchups
--------
baselines : AimAtGap (team 0) vs HardOffense (team 1)
genome    : ParameterizedStrategy(rank0_latest genome) vs SmackBall; exercises
            the np.random calls inside ParameterizedStrategy.choose_hit

Usage
-----
  python debug_scripts/golden_points.py --record
  python debug_scripts/golden_points.py --check
"""

import _path  # noqa: F401  (must precede project imports)

import argparse
import hashlib
import json
import os
import subprocess
import sys

import numpy as np

from evolution import ParameterizedStrategy, load_agent
from field import Field
from simulation import simulate_point
from strategy import AimAtGap, HardOffense, SmackBall

GOLDEN_PATH = "output/golden_points.json"
GENOME_PATH = "output/agents/rank0_latest.json"
N_SEEDS = 200


def _matchups() -> dict:
    genome = load_agent(GENOME_PATH)[0].genome
    return {
        "baselines": lambda: {0: AimAtGap(), 1: HardOffense()},
        "genome":    lambda: {0: ParameterizedStrategy(genome), 1: SmackBall()},
    }


def _digest(result) -> str:
    """Hash every recorded frame and action-log entry of a point."""
    h = hashlib.sha256()
    for f in result.frames:
        h.update(repr((f.tick, f.ball_x, f.ball_y, f.ball_vx, f.ball_vy,
                       f.rod_ys, f.rod_xs, f.rod_ctrl, f.ups, f.event)).encode())
    for a in result.action_log:
        h.update(repr(a).encode())
    return h.hexdigest()[:16]


def _play_all() -> dict:
    out = {}
    for name, make in _matchups().items():
        rows = []
        for seed in range(N_SEEDS):
            r = simulate_point(Field(), make(), kickoff_team=seed % 2, record=True,
                               seed=seed, collect_action_log=True)
            rows.append([r.winner, r.ticks, r.outcome, r.is_self_goal, _digest(r)])
        out[name] = rows
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--record", action="store_true")
    g.add_argument("--check", action="store_true")
    args = ap.parse_args()

    results = _play_all()

    if args.record:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True).stdout.strip()
        os.makedirs(os.path.dirname(GOLDEN_PATH), exist_ok=True)
        with open(GOLDEN_PATH, "w") as f:
            json.dump({"commit": commit, "results": results}, f)
        print(f"recorded {sum(len(v) for v in results.values())} points at {commit} -> {GOLDEN_PATH}")
        return 0

    with open(GOLDEN_PATH) as f:
        golden = json.load(f)
    bad = 0
    for name, rows in results.items():
        for seed, (got, want) in enumerate(zip(rows, golden["results"][name])):
            if got != want:
                bad += 1
                if bad <= 5:
                    print(f"MISMATCH {name} seed {seed}: got {got} want {want}")
    if bad:
        print(f"FAIL: {bad} mismatches (golden recorded at {golden['commit']})")
        return 1
    print(f"OK: all points match golden (recorded at {golden['commit']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
