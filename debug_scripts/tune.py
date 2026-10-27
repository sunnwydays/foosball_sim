"""tune.py -- realism metrics for tuning timing/speed constants.

Runs N points with play.py's teams and skills (kickoff alternates by seed, so
neither team gets it every time) and prints each metric next to its target
band from TARGETS. Any config constant can be overridden from the command line;
overrides are applied BEFORE the sim modules import, so constants bound as
default arguments (Rod.movement_speed, step_ball friction, ContactParams) pick
them up too.

Usage
-----
  python tune.py [N] [NAME=VALUE ...]
      e.g.  python tune.py 300 REACTION_TIME=0.12 HIT_SPEED=160
  python tune.py --sweep NAME=v1,v2,v3 [N] [NAME=VALUE ...]
      runs one subprocess per value in parallel and prints a comparison table

Results are deterministic: same code + same overrides = same numbers.
"""

import json
import math
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import _path  # noqa: F401  (project root on sys.path)
import config

# Target bands (lo, hi); None = unbounded. These are judgement calls about what
# real foosball looks like -- edit them as your idea of "realistic" changes.
TARGETS = {
    "whiff_slow":   (None, 0.05),   # ball < 30 cm/s: a near-stopped ball is almost never missed
    "whiff_mid":    (0.05, 0.15),   # 30-100 cm/s: passes, rebounds
    "whiff_fast":   (0.10, 0.30),   # >= 100 cm/s: shots are genuinely hard to time
    "own_goal":     (None, 0.10),   # share of goals scored into your own net
    "no_goal":      (None, 0.03),   # dead ball + stallout + timeout share of points
    "point_len":    (5.0, 15.0),    # seconds, mean
    "fwd_goals":    (0.40, None),   # share of goals from the forward (3-bar) rod
}

SPEED_BUCKETS = (("slow", 30.0), ("mid", 100.0), ("fast", math.inf))

# Constants derived from others in config.py: follow them unless overridden
DERIVED = {
    "COMMIT_COOLDOWN":   (("SWING_DURATION", "SWING_RECOVERY"),
                          lambda: config.SWING_DURATION + config.SWING_RECOVERY),
    "PASSTHROUGH_SPEED": (("BALL_MAX_SPEED",), lambda: config.BALL_MAX_SPEED * 0.9),
}


def parse_overrides(args: list[str]) -> dict[str, float]:
    out = {}
    for a in args:
        name, value = a.split("=", 1)
        if not hasattr(config, name):
            sys.exit(f"unknown config constant: {name}")
        out[name] = float(value)
    return out


def apply_overrides(overrides: dict[str, float]) -> None:
    for name, value in overrides.items():
        setattr(config, name, value)
    for name, (sources, fn) in DERIVED.items():
        if any(s in overrides for s in sources) and name not in overrides:
            setattr(config, name, fn())
    if "FPS" in overrides:
        config.DT = 1.0 / config.FPS
        config.MAX_TICKS = int(config.MAX_GAME_TIME * config.FPS)


def _bucket(speed: float) -> str:
    return next(name for name, hi in SPEED_BUCKETS if speed < hi)


def _last_touch_team(log) -> int | None:
    """
    Team of the last figure to touch the ball by any means: a HIT, a held-rod
    PASSIVE block, or a free-rod DEFLECT. A DEFLECT logs every rod in reach,
    not the one touched, so it only counts when those rods are all one team's
    (approximate). Whiffs pass under the man and are not touches.
    """
    for e in reversed(log):
        if e.action in ("HIT", "PASSIVE"):
            return e.team
        if e.action == "DEFLECT":
            teams = {int(label[1]) for label in e.rod_label.split(",") if label.startswith("T")}
            if len(teams) == 1:
                return teams.pop()
    return None


def measure(n: int) -> dict:
    import play
    from play import run_point, TEAM_0, TEAM_1, TEAM_0_SKILL, TEAM_1_SKILL

    hits = {b: 0 for b, _ in SPEED_BUCKETS}
    whiffs = {b: 0 for b, _ in SPEED_BUCKETS}
    outcomes: dict[str, int] = {}
    goals = own = own_touch = fwd = t0_wins = 0
    total_t = 0.0
    n_hits = 0
    for seed in range(n):
        _, res, _ = run_point(seed, TEAM_0, TEAM_1, TEAM_0_SKILL, TEAM_1_SKILL,
                              kickoff=seed % 2)
        outcomes[res.outcome] = outcomes.get(res.outcome, 0) + 1
        total_t += res.time
        t0_wins += res.winner == 0
        last_hit_label = {0: None, 1: None}
        for e in res.action_log:
            if e.action == "HIT":
                n_hits += 1
                last_hit_label[e.team] = e.rod_label
                # Incoming speed: the ball at the start of the hit tick
                f = res.frames[max(round(e.game_time / config.DT) - 1, 0)]
                hits[_bucket(math.hypot(f.ball_vx, f.ball_vy))] += 1
            elif e.action == "WHIFF":
                whiffs[_bucket(math.hypot(*e.actual_vel))] += 1
        if res.outcome == "goal":
            goals += 1
            own += bool(res.is_self_goal)
            own_touch += _last_touch_team(res.action_log) == 1 - res.winner
            label = last_hit_label[res.winner]
            fwd += label is not None and label.endswith("fwd") and not res.is_self_goal

    def rate(b):
        tries = hits[b] + whiffs[b]
        return whiffs[b] / tries if tries else 0.0

    return {
        "n": n,
        "whiff_slow": rate("slow"), "whiff_mid": rate("mid"), "whiff_fast": rate("fast"),
        "whiff_all": sum(whiffs.values()) / max(1, sum(whiffs.values()) + sum(hits.values())),
        "tries": {b: hits[b] + whiffs[b] for b, _ in SPEED_BUCKETS},
        "own_goal": own / max(1, goals),
        "own_touch": own_touch / max(1, goals),
        "no_goal": (n - goals) / n,
        "outcomes": outcomes,
        "point_len": total_t / n,
        "fwd_goals": fwd / max(1, goals),
        "hits_per_point": n_hits / n,
        "t0_win": t0_wins / n,
    }


def verdict(name: str, value: float) -> str:
    lo, hi = TARGETS[name]
    ok = (lo is None or value >= lo) and (hi is None or value <= hi)
    band = f"{'' if lo is None else lo}..{'' if hi is None else hi}"
    return f"{'ok ' if ok else 'OFF'} (target {band})"


def report(m: dict, overrides: dict[str, float]) -> None:
    print(f"n={m['n']}  overrides: {overrides or 'none'}")
    t = m["tries"]
    for name, label in (("whiff_slow", f"whiff <30 cm/s    ({t['slow']} tries)"),
                        ("whiff_mid",  f"whiff 30-100 cm/s ({t['mid']} tries)"),
                        ("whiff_fast", f"whiff >=100 cm/s  ({t['fast']} tries)")):
        print(f"  {label:<32} {m[name]:6.1%}  {verdict(name, m[name])}")
    print(f"  {'whiff overall':<32} {m['whiff_all']:6.1%}")
    print(f"  {'own goals (of goals)':<32} {m['own_goal']:6.1%}  {verdict('own_goal', m['own_goal'])}")
    print(f"  {'  last touch by conceder, any':<32} {m['own_touch']:6.1%}")
    print(f"  {'no-goal points':<32} {m['no_goal']:6.1%}  {verdict('no_goal', m['no_goal'])}  {m['outcomes']}")
    print(f"  {'mean point length':<32} {m['point_len']:5.2f}s  {verdict('point_len', m['point_len'])}")
    print(f"  {'goals from forward rod':<32} {m['fwd_goals']:6.1%}  {verdict('fwd_goals', m['fwd_goals'])}")
    print(f"  {'hits per point':<32} {m['hits_per_point']:6.1f}")
    print(f"  {'team 0 win rate':<32} {m['t0_win']:6.1%}")


def sweep(spec: str, rest: list[str]) -> None:
    name, values = spec.split("=", 1)
    n_args = [a for a in rest if "=" not in a]
    fixed = [a for a in rest if "=" in a]

    def run(v: str) -> dict:
        cmd = [sys.executable, __file__, "--json", *n_args, *fixed, f"{name}={v}"]
        out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
        return json.loads(out.strip().splitlines()[-1])

    values = values.split(",")
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as pool:
        results = list(pool.map(run, values))

    cols = ["whiff_slow", "whiff_mid", "whiff_fast", "own_goal", "own_touch", "no_goal",
            "point_len", "fwd_goals", "hits_per_point", "t0_win"]
    print(f"sweep {name}  (fixed: {' '.join(fixed) or 'none'})")
    print(f"{name:>16} " + " ".join(f"{c:>11}" for c in cols))
    for v, m in zip(values, results):
        cells = []
        for c in cols:
            s = f"{m[c]:.2f}" if c in ("point_len", "hits_per_point") else f"{m[c]:.1%}"
            if c in TARGETS and "OFF" in verdict(c, m[c]):
                s = "*" + s
            cells.append(f"{s:>11}")
        print(f"{v:>16} " + " ".join(cells))
    print("* = outside its target band")


def main() -> None:
    args = sys.argv[1:]
    if args and args[0] == "--sweep":
        sweep(args[1], args[2:])
        return
    as_json = bool(args) and args[0] == "--json"
    if as_json:
        args = args[1:]
    n = int(args[0]) if args and "=" not in args[0] else 300
    overrides = parse_overrides([a for a in args if "=" in a])
    apply_overrides(overrides)   # before import: default arguments bind these
    import play  # noqa: F401  (sets FPS/DT from play.py)
    apply_overrides(overrides)   # after import: so an FPS override survives play.py
    m = measure(n)
    if as_json:
        print(json.dumps(m))
    else:
        report(m, overrides)


if __name__ == "__main__":
    main()
