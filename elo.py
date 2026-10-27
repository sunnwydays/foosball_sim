"""
elo.py -- rank every agent on an Elo-scale ladder.

Every pair plays a fixed number of points (half with each agent as team 0), then
ratings come from a Bradley-Terry maximum-likelihood fit: P(i beats j) =
p_i / (p_i + p_j), solved with the MM update

    p_i  <-  W_i / sum_j n_ij / (p_i + p_j)        W_i = wins + 0.5 * draws

and reported on the Elo scale, elo_i = 400 * log10(p_i), mean anchored at 1500.
Unlike sequential Elo updates the fit does not depend on game order. 95% CIs
come from bootstrapping the pairwise results.

Usage
-----
  python elo.py --ppo vs_pool=output/ppo/ppo_vs_pool.zip \
                --ppo selfplay=output/ppo/ppo_selfplay.zip [--points 400]
  python elo.py --rerate        # refit the saved results, no new games

Outputs: output/ladder.csv, output/ladder.png, output/ladder_results.json
"""

import argparse
import csv
import json
import math
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations

import numpy as np

GA_PATH = "output/agents/rank0_latest.json"
RESULTS_PATH = "output/ladder_results.json"
CHUNK = 100   # points per work item (even: half on each side)


# ---------------------------------------------------------------------------
# Playing
# ---------------------------------------------------------------------------

def _build(spec: str):
    """spec: 'baseline:<name>' | 'ga:<path>' | 'ppo:<path>'"""
    from evolution import BASELINES, ParameterizedStrategy, load_agent
    from ppo_strategy import PPOStrategy, load_body_genome
    kind, _, arg = spec.partition(":")
    if kind == "baseline":
        return BASELINES[arg]()
    if kind == "ga":
        return ParameterizedStrategy(load_agent(arg)[0].genome)
    return PPOStrategy.from_file(arg, load_body_genome())


def _worker(args) -> tuple[int, int, int]:
    """(spec_a, spec_b, n_points, seed0) -> (a wins, b wins, draws). Half the points on each side."""
    spec_a, spec_b, n, seed0 = args
    from field import Field
    from ppo_strategy import play_point
    a, b = _build(spec_a), _build(spec_b)
    wa = wb = d = 0
    for i in range(n):
        side_a = i % 2
        r = play_point(Field(), {side_a: a, 1 - side_a: b},
                       kickoff_team=(i // 2) % 2, seed=seed0 + i)
        if r.winner is None:
            d += 1
        elif r.winner == side_a:
            wa += 1
        else:
            wb += 1
    return wa, wb, d


def play_ladder(agents: dict[str, str], points: int, workers: int | None = None) -> dict:
    """Round robin: {'a|b': [a_wins, b_wins, draws]} for every pair."""
    chunk = min(CHUNK, points)
    jobs, keys = [], []
    for (na, sa), (nb, sb) in combinations(agents.items(), 2):
        for c in range(math.ceil(points / chunk)):
            jobs.append((sa, sb, chunk, 10_000 * len(jobs)))
            keys.append(f"{na}|{nb}")
    results: dict[str, list[int]] = {}
    with ProcessPoolExecutor(workers) as pool:
        for key, (wa, wb, d) in zip(keys, pool.map(_worker, jobs)):
            r = results.setdefault(key, [0, 0, 0])
            r[0] += wa; r[1] += wb; r[2] += d
    return results


# ---------------------------------------------------------------------------
# Rating
# ---------------------------------------------------------------------------

def fit_bt(names: list[str], results: dict, iters: int = 2000) -> np.ndarray:
    """Bradley-Terry MM fit -> Elo ratings (mean 1500), in `names` order."""
    n = len(names)
    idx = {name: i for i, name in enumerate(names)}
    games = np.zeros((n, n))
    wins = np.zeros(n)
    for key, (wa, wb, d) in results.items():
        a, b = (idx[x] for x in key.split("|"))
        games[a, b] = games[b, a] = wa + wb + d
        wins[a] += wa + 0.5 * d
        wins[b] += wb + 0.5 * d
    wins = np.maximum(wins, 0.5)   # a 0-win agent would otherwise get rating -inf
    p = np.ones(n)
    for _ in range(iters):
        denom = (games / (p[:, None] + p[None, :])).sum(axis=1)
        new = wins / denom
        new /= np.exp(np.log(new).mean())   # geometric-mean anchor: BT is scale-free
        if np.abs(new - p).max() < 1e-10:
            p = new
            break
        p = new
    return 1500.0 + 400.0 * np.log10(p)


def bootstrap(names: list[str], results: dict, n_boot: int = 200, seed: int = 0) -> np.ndarray:
    """(n_boot, n_agents) Elo samples from resampling each pair's points."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        resampled = {}
        for key, (wa, wb, d) in results.items():
            tot = wa + wb + d
            resampled[key] = rng.multinomial(tot, np.array([wa, wb, d]) / tot).tolist()
        out.append(fit_bt(names, resampled))
    return np.array(out)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def report(names: list[str], results: dict, commit: str) -> None:
    elo = fit_bt(names, results)
    samples = bootstrap(names, results)
    lo, hi = np.percentile(samples, [2.5, 97.5], axis=0)
    order = np.argsort(-elo)

    print(f"physics commit {commit}")
    print(f"{'rank':>4}  {'agent':<18}{'elo':>7}   95% CI")
    with open("output/ladder.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "agent", "elo", "ci_low", "ci_high", "commit"])
        for rank, i in enumerate(order, 1):
            print(f"{rank:>4}  {names[i]:<18}{elo[i]:>7.0f}   [{lo[i]:.0f}, {hi[i]:.0f}]")
            w.writerow([rank, names[i], f"{elo[i]:.1f}", f"{lo[i]:.1f}", f"{hi[i]:.1f}", commit])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(names)
    wr = np.full((n, n), np.nan)
    idx = {name: i for i, name in enumerate(names)}
    for key, (wa, wb, d) in results.items():
        a, b = (idx[x] for x in key.split("|"))
        tot = wa + wb + d
        wr[a, b] = (wa + 0.5 * d) / tot
        wr[b, a] = (wb + 0.5 * d) / tot

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ys = np.arange(n)[::-1]
    ordered = list(order)
    ax1.barh(ys, elo[ordered], xerr=[elo[ordered] - lo[ordered], hi[ordered] - elo[ordered]], color="#4c78a8")
    ax1.set_yticks(ys, [names[i] for i in ordered])
    ax1.axvline(1500, color="gray", lw=0.8)
    ax1.set_xlabel("Elo (Bradley-Terry, mean 1500, 95% bootstrap CI)")
    ax1.set_title(f"Ladder  (physics {commit})")

    im = ax2.imshow(wr[np.ix_(ordered, ordered)], cmap="RdBu", vmin=0.2, vmax=0.8)
    ax2.set_xticks(range(n), [names[i] for i in ordered], rotation=60, ha="right")
    ax2.set_yticks(range(n), [names[i] for i in ordered])
    ax2.set_title("Row's score vs column (win = 1, draw = 0.5)")
    fig.colorbar(im, ax=ax2)
    fig.tight_layout()
    fig.savefig("output/ladder.png", dpi=120)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ppo", action="append", default=[], metavar="NAME=PATH")
    ap.add_argument("--points", type=int, default=400, help="points per pair")
    ap.add_argument("--rerate", action="store_true", help="refit saved results only")
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()

    if args.rerate:
        with open(RESULTS_PATH) as f:
            saved = json.load(f)
        report(saved["agents"], saved["results"], saved["commit"])
        return

    from evolution import BASELINES
    agents = {name: f"baseline:{name}" for name in BASELINES}
    agents["GA-rank0"] = f"ga:{GA_PATH}"
    for item in args.ppo:
        name, _, path = item.partition("=")
        agents[f"PPO-{name}"] = f"ppo:{path}"

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    if subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                      capture_output=True, text=True).stdout.strip():
        commit += "+uncommitted"
    results = play_ladder(agents, args.points, args.workers)
    os.makedirs("output", exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump({"commit": commit, "agents": list(agents), "points": args.points,
                   "results": results}, f, indent=1)
    report(list(agents), results, commit)


if __name__ == "__main__":
    main()
