# PPO / Gymnasium / Elo notes (branch `ppo-gym-elo`, built 2026-10-27)

Plan that drove this: `docs/superpowers/plans/2026-10-27-ppo-gym-elo.md`.
Everything under `output*/` is gitignored, so results and models are recorded here.

## What exists

| File | Role |
|---|---|
| `simulation.py` | `iter_point` generator yields a `TickView` before tick 0 and after each tick; `simulate_point` drains it |
| `observation.py` | `build_obs(view, team)`: 47 floats, mirrored for team 1 so one net plays both sides |
| `ppo_strategy.py` | `PPOStrategy` = GA body + learned `choose_hit`; `play_point` drives `iter_point` and lets model strategies `observe` each tick |
| `foosball_env.py` | `FoosballEnv`: episode = one point, step = one tick, `MultiDiscrete([6,3,2])` = aim x speed x commit, reward +1/-1/0 |
| `train_ppo.py` | SB3 PPO on 12 `SubprocVecEnv` workers; eval CSV every 100k steps; `--selfplay` snapshots |
| `elo.py` | round robin + Bradley-Terry MLE on Elo scale, bootstrap CIs |
| `debug_scripts/golden_points.py` | `--record` / `--check`: exact-output regression of `simulate_point` over 400 points |
| `evolution.py` | only change: `_aim_target(choice, ...)` extracted so GA and PPO share aim primitives |

## Concepts mapped to the code

- **GA genome vs PPO policy.** A genome is a fixed mixture over primitives (e.g. `shot_rate`,
  `aim_gap/wall/player`); the same probabilities apply in every state. The PPO policy is a network
  `obs -> distribution over (aim, speed, commit)`, so the choice depends on state. PPO keeps the GA
  champion's body (hands, positioning, passive rods, skills), so rating gaps come from `choose_hit` only.
- **Episode / step / reward.** One point per episode, one tick per step, +1 scored, -1 conceded, 0
  otherwise. No shaping (a possession-stall rule is an obvious loophole). `timeout` is `truncated`
  so PPO bootstraps V(s); goal / stallout / dead_ball are `terminated`.
- **Generator = control inversion.** The tick loop stays untouched; the trainer resumes it one tick at
  a time and reads state between ticks. RNG call order is unchanged, which is why the golden check matches exactly.
- **PPO pieces.** Advantage = how much better an action was than the value baseline V(s) predicts
  (GAE, `gae_lambda=0.95`, `gamma=0.995`). The clipped ratio (`clip_range=0.2`) is a cheap trust region: updates
  stop pushing once the new policy deviates ~20% from the one that collected the data. The value net is a
  variance-reduction baseline, the same role many round-robin points play in the GA's fitness.
  `ent_coef=0.01` keeps the policy stochastic (a mixed strategy on purpose).
- **Self-play.** Opponent pool sampled per `reset()`: 6 baselines + GA + frozen PPO snapshots (50% snapshots once any exist).
  Same idea as the GA Hall of Fame. From SB3's view it is still single-agent.
- **Mirroring.** Rod x positions are symmetric about 60, so team 1 sees `x -> 120 - x`, `vx -> -vx`,
  `x_offset -> -x_offset`. Actions need no mirroring (aim helpers are team-aware).
- **Bradley-Terry vs incremental Elo.** BT fits all results at once (MM update, order independent) and
  gives CIs by bootstrapping; sequential Elo depends on game order. Elo scale = `400*log10(p)`, mean 1500.

## Results (physics commit 650fec9, GA body = the bugged-run genome below)

Training speed: about 700-900 steps/s on 12 workers with 16 cores (plan assumed 3-6k).
Mean win rate vs the 7 fixed opponents (6 baselines + GA, 50 pts each, draw = 0.5):

| Steps | 0 | 200k | 600k | 1.5M | 2.0M (end of phase A) |
|---|---|---|---|---|---|
| Mean | 0.36 | 0.38 | 0.47 | 0.49 | 0.47 |

Plateaued from about 1.5M; per-point eval noise is about +-0.03. Phase B (500k steps, snapshots every 125k)
stayed at 0.44-0.51.

Ladder (400 pts per pair, half on each side, 95% bootstrap CI):

| Rank | Agent | Elo | CI |
|---|---|---|---|
| 1 | DefensiveWall | 1545 | 1535-1554 |
| 2 | TiltAndGap | 1545 | 1537-1553 |
| 3 | ReactiveBlock | 1520 | 1511-1529 |
| 4 | GA-rank0 | 1508 | 1498-1518 |
| 5 | PPO-vs_pool | 1490 | 1482-1498 |
| 6 | HardOffense | 1489 | 1480-1500 |
| 7 | PPO-snap2.25M | 1485 | 1477-1493 |
| 8 | SmackBall | 1477 | 1467-1487 |
| 9 | PPO-selfplay | 1473 | 1465-1482 |
| 10 | AimAtGap | 1468 | 1459-1478 |

Honest reading: PPO sits mid-ladder, about on par with HardOffense and SmackBall, below the GA champion and the
top three baselines. Self-play gave no gain (1473 vs 1490, CIs nearly touch). Resume bullet 3 is true as worded
(PPO self-play agents, custom Gymnasium env, Elo ladder); do not claim PPO beat the GA or the strong baselines.

## Caveats

- The GA body is `output/agents/rank0_latest.json`, the rank 0 of a **bugged** evolution run (win_rate 0.6135). Genome:

```json
[0.13081652991313414, 0.14094535907526912, 0.7282381110115967, 0.3633450400856206,
 0.6366549599143795, 0.0, 0.9785835212384358, 0.02141647876156412, 0.0, 0.0, 1.0, 1.0,
 0.7784493760019736, 0.9358354476603011, 0.8761694457910512, 0.9052835662229792,
 0.9928905207725914]
```
  Save that to `output/agents/rank0_latest.json` as `{"genome": [...]}` on a new machine, or point
  `GA_CHAMPION_PATH` in `ppo_strategy.py` at the fresh evolution output. `PPOStrategy` falls back to a neutral genome if the file is missing.
- If the fresh (fixed-physics) GA run finishes: rerun phase A (about 500k steps is probably enough), then the ladder.
  GA, PPO and the ladder must share one physics commit.
- `output/golden_points.json` was recorded at 2811077; re-record with `golden_points.py --record` after any intended physics change.
- Models (`output/ppo/*.zip`) are not in git. Regenerate with the commands below.

## Reproduce

```
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-rl.txt --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe train_ppo.py --steps 2000000 --out output/ppo/ppo_vs_pool
.\.venv\Scripts\python.exe train_ppo.py --steps 500000 --init output/ppo/ppo_vs_pool.zip --selfplay --snap-every 125000 --out output/ppo/ppo_selfplay
.\.venv\Scripts\python.exe elo.py --points 400 --ppo vs_pool=output/ppo/ppo_vs_pool.zip --ppo selfplay=output/ppo/ppo_selfplay.zip
```

Phase B resume note: `num_timesteps` continues from the loaded model, so callbacks count from the start step.
`<out>_latest.zip` is saved at every eval, so a run can be stopped early and keep its model.

## Ideas not done

- Defense-mode action dimension; a CleanRL-style single-file PPO for understanding; README section + replay GIF.
- Flat eval curve after 1.5M: try bigger batch / higher `ent_coef`, richer observation (predicted contact time), or per-rod actions.
- Throughput (700-900 steps/s) is below the 5,000 ticks/s single-core sim rate; worth profiling `PPOStrategy` / IPC overhead.
