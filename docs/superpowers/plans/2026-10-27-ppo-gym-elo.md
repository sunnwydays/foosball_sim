# Plan: Gymnasium env + PPO self-play + ELO ladder (build 2026-10-27 morning)

Goal: make resume bullet 3 true and explainable before the interview:
"PPO self-play agents in a custom Gymnasium env, all agents ranked on an ELO ladder."
Bullet 3 does not claim PPO wins. A mid-ladder PPO agent still makes it true; report honestly.

## Decisions (locked 2026-10-26)

| Decision | Choice | Why |
|---|---|---|
| Library | Gymnasium 1.4 + stable-baselines3 2.9 + torch CPU | OpenAI Gym is unmaintained; SB3 needs Gymnasium. Net is tiny, sim is the bottleneck, GPU useless |
| Control inversion | **Generator**: `iter_point()` yields once per tick; `simulate_point()` drains it | ~15-line diff, RNG call order untouched, so determinism is preserved by construction. Smaller than a `PointSim` class refactor |
| Episode | One point | Matches "each point is independent" |
| Step | One tick (`config.DT`) | Measured 5,000 ticks/s, 53 points/s, ~95 ticks/point on one core (AimAtGap vs HardOffense, FPS 10) |
| Action | `MultiDiscrete([6, 3, 2])` = aim x speed x commit | Same primitives as the GA genome, so GA vs PPO isolates the learning method |
| Body (hands/pos/passive/skill) | Inherited from GA champion genome (`output/agents/rank0.json`); fallback neutral genome | PPO agent = GA champion with a state-conditioned `choose_hit`. ELO gap between them is attributable to the policy alone. Same skill budget (skill group sums to 1) |
| Reward | +1 point won, -1 point lost, 0 otherwise. No shaping | Shaping invites hacking (possession-stall rule is an obvious loophole) |
| Termination | `goal`, `stallout`, `dead_ball` -> `terminated`; `timeout` -> `truncated` | Truncated lets PPO bootstrap V(s) instead of treating time-up as a real ending |
| Self-play | Opponent pool sampled per `reset()`: baselines + GA rank0 + frozen PPO snapshots | From SB3's view it is single-agent with stochastic dynamics. Same idea as the GA Hall of Fame |
| Ratings | Bradley-Terry MLE on round-robin results, reported on Elo scale, draws = 0.5 | Order-independent, unlike sequential Elo updates |

### Action mapping (`PPOStrategy.choose_hit`)

- `aim` 0..5: gap / off wall / off player / pass forward / pass back / pass side. Same fallbacks as
  `ParameterizedStrategy.choose_hit` (off-player with no passive opp rod -> gap; missing pass rod -> goal center).
- `speed` 0..2: `config.HIT_SPEED * (0.5 + 0.5 * s)`, `s in {0, 0.5, 1}` (same formula as `shot_speed` gene).
- `commit` 0..1: 0 returns `None` (hold), 1 calls `_commit_swing(...)`, which still owns timing.
- Must still set `rod.accuracy / power_consistency / anticipation` from the body genome, as the GA's `choose_hit` does.
- Extract the aim-choice block of `ParameterizedStrategy.choose_hit` into a shared `_aim_target(choice, rod, ball, field)`
  so there is one source of truth. Keep the `np.random` calls in the GA path in the same order (golden check covers it).

### Observation (~46 floats, roughly [-1, 1])

- Ball: x, y, vx, vy (positions / field dims, velocities / `BALL_MAX_SPEED`).
- 8 rods, own team first then opponent, each ordered from own goal outward:
  y_offset / slide half-range, x_offset / `ROD_X_REACH`, up, controlled, swing armed.
- Own `reacting` flag, possession timers / `POSSESSION_LIMIT`.
- **Mirror for team 1**: x -> `FIELD_DEPTH - x`, vx -> -vx, x_offset sign flipped. Rod x positions are symmetric about 60,
  so the agent always "attacks toward +x" and one policy plays both sides. Actions need no mirroring (aim helpers are already team-aware).

## Setup (5 min, can be done tonight)

Use the project venv (Python 3.14). Bare `pip` points at a different Python 3.10 install.

```
.\.venv\Scripts\python.exe -m pip install gymnasium stable-baselines3 torch --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
```

Dry run on 2026-10-26 resolved torch 2.14.1+cpu, gymnasium 1.4.0, stable_baselines3 2.9.2.
Add `requirements-rl.txt` (`-r requirements.txt`, gymnasium, stable-baselines3, torch).

## Step 0: freeze physics, start the GA (T+0:00)

- The bug investigation (point length 5.7s -> 13.2s, skill inversion) must be merged or shelved **before** the GA run.
  GA, PPO and the ladder must all run on one physics commit, or the ladder compares agents from different games.
- Record that commit hash in the ladder output.
- Start `python evolution.py` (other machine if available; else here in the background, ~1M points,
  est. 35-45 min on 12 cores). Its end-of-run benchmark (500 pts vs each baseline) is the bullet-2 check.
  If HardOffense is still a tie, rephrase bullet 2 as "beat 5 of 6, tied the strongest".

## Step 1: generator refactor (T+0:00 to 0:20)

1. Before touching `simulation.py`: write `debug_scripts/golden_points.py --record/--check`. For seeds 0..199 on 2 matchups at
   default FPS, store `(winner, ticks, outcome, is_self_goal, hash of frame ball positions)` to `output/golden_points.json`.
   Record on the frozen commit.
2. In `simulation.py`: rename body to `iter_point(...)` (same signature). `yield` once after pre-game setup and once at the end
   of each tick (after frame record). Each `return PointResult(...)` stays (becomes `StopIteration.value`).
   Yield a small view `(ball, field, team_states, possession_timer, game_time)` for the env to build obs.
3. `simulate_point(...)` = create generator, drain it, return `StopIteration.value`. Public API unchanged.
4. Verify: `golden_points.py --check` exact match, `pytest`, `pytest -m slow`.

Note: config is monkeypatched in tests (FPS 30), so keep reading `config.DT` / `MAX_TICKS` at call time, not import time.

## Step 2: env + strategy adapter (T+0:20 to 1:10)

`foosball_env.py`
- `build_obs(view, team) -> np.ndarray` (with mirroring), shared by env and `PPOStrategy`.
- `FoosballEnv(gym.Env)`: `observation_space = Box(-1, 1, (N,), float32)` (clip obs), `action_space = MultiDiscrete([6, 3, 2])`.
- `reset(seed)`: pick `agent_team` in {0, 1} and kickoff at random, sample opponent from pool, create `iter_point` generator,
  advance to the first yield, return obs.
- `step(action)`: set `agent_strategy.intent = action`, `next(gen)`; on `StopIteration` read the `PointResult`, compute reward,
  set terminated/truncated, put `outcome` and `is_self_goal` in `info`.
- `add_opponent(path)`: append a PPO snapshot to the pool (called via `VecEnv.env_method` from training).
- Opponent sampling: 50% {6 baselines + GA rank0}, 50% PPO snapshots (until snapshots exist, 100% fixed pool).

`ppo_strategy.py`
- `PPOStrategy(ParameterizedStrategy)`: body genome from rank0 or neutral. Two modes:
  - training agent: `intent` set externally by the env each tick.
  - frozen opponent / ladder: holds an SB3 model, computes `build_obs` for its own team, samples an action (stochastic;
    the policy is a mixed strategy on purpose). Lazy-load model with `device="cpu"`.
- Call `torch.set_num_threads(1)` in workers: 12 subprocesses x 12 torch threads oversubscribes the CPU.

`tests/test_env.py` (fast)
- `stable_baselines3.common.env_checker.check_env(FoosballEnv())` passes.
- Random-action episodes always end; rewards in {-1, 0, 1}; obs within bounds.
- Mirroring: obs for team 1 equals obs for team 0 on the x-mirrored state.

## Step 3: training (T+1:10 to 2:30)

`train_ppo.py` (`if __name__ == "__main__":` guard, required for Windows spawn)
- `SubprocVecEnv` x 12, `VecMonitor`.
- PPO(`MlpPolicy`, net [64, 64], `n_steps=512` (batch 6,144), `batch_size=1024`, `n_epochs=10`, `gamma=0.995`,
  `gae_lambda=0.95`, `clip_range=0.2`, `ent_coef=0.01`, `lr=3e-4`). gamma: 0.99^95 = 0.38 vs 0.995^95 = 0.62 credit at kickoff.
- Callbacks:
  - every 100k steps: 50 pts vs each fixed opponent -> `output/ppo/eval.csv` (this is the learning-curve plot for the interview).
  - self-play phase: every 250k steps save `output/ppo/snap_<steps>.zip` and `env_method("add_opponent", path)`.
- Runs:
  1. Smoke: 50k steps. Measure steps/s (estimate 3-6k/s, unverified). Confirm eval.csv is written.
  2. Phase A: 2M steps vs fixed pool -> `ppo_vs_pool.zip`.
  3. Phase B: continue from A for 1-2M steps with self-play snapshots on -> `ppo_selfplay.zip`.
- Start Phase A in the background, write `elo.py` while it runs.

## Step 4: ELO ladder (T+2:00 to 3:00)

`elo.py`
- Agents: 6 baselines, GA rank0 (+ rank1 optional), PPO `vs_pool`, PPO `selfplay`, 1-2 early snapshots (shows progression).
- Every pair, 400 pts, half with each agent as team 0 (side-swap removes any team-0 bias). Parallel via `ProcessPoolExecutor`
  like `evolution._make_pool`.
- Bradley-Terry MM update: `p_i <- W_i / sum_j n_ij / (p_i + p_j)`, `W_i = wins + 0.5 * draws`; normalize; Elo = `400*log10(p_i)`,
  anchored so mean = 1500. Bootstrap 200 resamples for 95% CIs.
- Outputs: `output/ladder.csv`, `output/ladder.png` (ranked bars with CI whiskers), pairwise win-rate heatmap, physics commit hash.
- ~12 agents -> 66 pairs x 400 = 26k points, est. a few minutes on 12 cores.

## Cut lines (if the morning runs short)

- **MVP**: Steps 1-2, Phase A only, ladder with baselines + GA + PPO. Bullet 3 true except "self-play".
- **MVP + self-play**: add Phase B (even 500k steps). Bullet 3 fully true.
- **Stretch**: defense-mode action dim, CleanRL-style single-file PPO for understanding, README section + replay GIF of PPO agent.

## Risks

- Bug fix lands mid-morning and changes physics: re-run GA + PPO or leave the fix for after the interview. Decide in Step 0.
- `simulation.py` merge conflict with the bug branch: refactor after it merges; generator diff is small.
- PPO fails to learn (flat eval curve): check reward sign per team and mirroring first (most likely bugs), then raise `ent_coef`
  or batch size. Sim noise is high, so expect slow, noisy curves.
- Torch thread oversubscription in subprocesses: `torch.set_num_threads(1)`.

## Interview talking points to prepare

- GA vs PPO: genome = fixed mixture over primitives, fitness = RR win rate; PPO = state-conditioned policy, gradient on per-decision advantage.
- PPO clip as a cheap trust region; value baseline as variance reduction (same reason the GA needs many RR points).
- Generator as control inversion: the trainer drives the sim one tick at a time without rewriting the tick loop.
- Self-play pool = GA Hall of Fame idea; mixed strategies = entropy, links to Phase 3 game theory.
- Bradley-Terry vs incremental Elo (relevant to Chess.com).
