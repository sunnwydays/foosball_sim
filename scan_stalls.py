"""scan_stalls.py -- find seeds where a point ends by possession limit
(winner set but no goal event) or runs long with few hits."""

import config
config.FPS = 30
config.DT = 1.0 / 30
config.MAX_TICKS = int(config.MAX_GAME_TIME * 30)

from field import Field
from strategy import HardOffense, TiltAndGap
from simulation import simulate_point

for seed in range(300):
    field = Field()
    for rod in field.rods:
        if rod.team == 0:
            rod.accuracy, rod.power_consistency = (0.7, 0.7)
        else:
            rod.accuracy, rod.power_consistency = (0.6, 0.7)
    strats = {0: HardOffense(), 1: TiltAndGap()}
    result = simulate_point(field, strats, kickoff_team=1, record=True,
                            seed=seed, collect_action_log=True)
    hits = sum(1 for f in result.frames if f.event == "hit")
    goal = any(f.event and f.event.startswith("goal") for f in result.frames)
    poss_limit = result.winner is not None and not goal
    suspicious = (result.time > 6.0 and hits < 12)
    if poss_limit or suspicious or result.winner is None:
        why = []
        if poss_limit: why.append("POSSESSION-LIMIT")
        if result.winner is None: why.append("DRAW/TIMEOUT")
        if suspicious: why.append("LONG-FEW-HITS")
        retracts = sum(1 for e in result.action_log if e.action == "RETRACT")
        print(f"seed={seed:3d} winner={result.winner} t={result.time:6.2f}s "
              f"hits={hits:3d} retracts={retracts:3d}  {','.join(why)}")
