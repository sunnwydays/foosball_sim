# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Intent

A probabilistic foosball simulator in Python, built as a learning project for Monte Carlo simulation, with the long-term goal of scaling to other sports (soccer, basketball). The model is **probability-based, not physics-based** — each action (hit, pass, shot) has an associated probability distribution rather than physical equations of motion.

## Core Design Principles

- **Parameterize everything** — skill level, consistency, field dimensions, rod positions, shot distributions should all be configurable, not hardcoded
- **Each point is independent** — a single possession/point is the atomic simulation unit; win probability comes from running many independent trials
- **Probability distributions over physics** — direction and speed follow normal distributions; "skill" narrows the distribution (less variance), "accuracy" shifts the mean toward the intended target
- **No hand-switching constraint** — any rod can be controlled at any time (simplification)

## Finalized Design Decisions

These were locked in before the first implementation and should not be changed without good reason:

- **Tick-based simulation** — both teams make decisions every tick (10 FPS); `REACTION_TIME=0.3s` lockout prevents changing rod targets immediately after the opponent hits; no "turns"
- **Ball physics** — time-stepped with substepping; wall bounces (y-component flips on side wall contact); no out-of-bounds
- **Swing window model** — a controlled rod commits a swing in advance; the swing connects only if the ball arrives within the committed active window (`BACKSWING_RATIO * SWING_DURATION` to `SWING_DURATION` after commit); a mistimed swing or unarmed rod rigid-bounces the ball at `CONTROLLED_SLOWDOWN=0.6x`; this replaces the old "possession transfer" model
- **Shot model** — attacker chooses a target angle; actual angle is sampled from `Normal(target, 1/skill)`; speed sampled from `Normal(intended_speed, 1/consistency)`
- **Interception** — defender pre-positions via `_predicted_y()` (friction-aware TTC, side-wall bounce folding, up to 4 end-wall x-bounces); a committed swing window must bracket the ball's actual arrival to connect; undefended contact bounces at `CONTROLLED_SLOWDOWN=0.6x`
- **3-goalie rod** — each side has a 3-player goalie rod spanning the full field width; eliminates edge/corner ramp complexity
- **Infinite bounce guard** — max bounce/step limit per point; if hit, the point resets (dead ball)
- **Ball reach = ball edge** — all contact and reach checks use `center ± BALL_RADIUS`, not just center
- **Draw condition = x-reach only** — `_teams_with_reach` checks rod x-proximity only; y-overlap is excluded because a rod can always slide to cover any ball y position

## Model Architecture

Three nested loops:

1. **Game loop** — game theory, opponent modeling, mixed strategies, score-state policy *(planned)*
2. **Point loop** — physics + strategy per point (`simulation.py`)
3. **Shot resolution** — Monte Carlo simulator (`monte_carlo.py`) *(complete)*

### Files

| File | Role |
|------|------|
| `config.py` | All constants (dimensions, physics, timing, noise ranges) |
| `field.py` | `Rod`, `Goal`, `BallState`, `TeamState`, `Field`, `SwingCommitment` |
| `physics.py` | `step_ball()` with substepping, passive contact, `find_overlapping_players()` |
| `simulation.py` | `simulate_point()` — one full point, tick loop, draw/possession checks |
| `monte_carlo.py` | `run_monte_carlo()` — N independent points, aggregated `MCResult` |
| `strategy.py` | `Strategy` ABC + 6 concrete strategies; prediction + geometry helpers: `_predict_contact`, `_best_wall_shot`, `_best_player_deflection` |
| `evolution.py` | Genetic algorithm: `Agent`, `ParameterizedStrategy`, `rr_tourney()`, `main()` |
| `visualization.py` | `draw_field()`, `draw_stats()`, `replay_point()`, `plot_evolution_stats()` |
| `play.py` | Dev tool — edit settings at top, run `python play.py` for single-point replay + heatmap |
| `main.py` | Experiment runner — 5,000-sim matchups between hand-coded strategies |

### Three Separate Reach-Check Functions

These are independent and serve different purposes — do not conflate them:

- `_teams_with_reach` (`simulation.py`) — draw/possession timer check; **x-reach only**
- `find_overlapping_players` (`physics.py`) — controlled hit detection; uses full 2D bounding box + ball radius

## Genome Schema (GENOME_SIZE = 17)

Four groups in `GENE_GROUPS` (evolution.py). Three normalized (Dirichlet init, sum=1, renormalized after mutation), one independent ([0,1]):

| Group | Genes | Mutation std |
|-------|-------|--------------|
| `skill` | accuracy, power_consistency, movement_control | 0.06 |
| `shot` | aim_at_gap, aim_off_wall, aim_off_player | 0.13 |
| `pass` | pass_forward, pass_back, pass_side | 0.13 |
| `indep` | offensive_bias, shot_rate, shot_speed, lift_attackers, passive_x_offset_attack, passive_x_offset_defense, defensive_activity, anticipation | 0.08 |

`aggression` was split into `offensive_bias` (rod selection), `shot_rate` (shoot vs pass), `shot_speed` (speed multiplier) to eliminate pleiotropy — one gene controlling three independent behavioral axes caused evolution to always converge to high aggression.

## Evolution Loop Parameters

```python
POP_SIZE=50, N_GENERATIONS=50, N_ELITE=2
RR_POINTS=16, BASE_POINTS=32, HOF_SAMPLE=6, HOF_POINTS=16
FITNESS_WEIGHTS={"rr": 1.0, "base": 1.0, "hof": 1.0}
GEN_BENCH_POINTS=40, BENCHMARK_POINTS=500
TOURNAMENT_SIZE=3, MUTATION_RATE=0.2
FROZEN_GENES={"shot_speed": 1.0, "anticipation": 1.0}
```

Fitness = round-robin + fixed hardcoded baselines + sampled Hall of Fame champions. Latest full run: `output/run3/` (80 generations).

After each run: top 5 agents saved to `output/agents/rank{0..4}.json`, benchmark vs all 6 hardcoded strategies printed, `output/evolution_stats.png` generated.

Run: `python evolution.py` | Profile (fast, small): `python evolution.py --profile`

## Roadmap

### Phase 2 — Genetic Algorithm *(active)*

Implemented: genome decoder, RR tournament with multiprocessing, elitism, crossover, per-group mutation, EvolutionHistory, visualization, end-of-run hardcoded benchmark, Hall of Fame + baseline fitness terms, Tier 1 hot-path speedup.

Remaining (in priority order):
1. **Gene modelling** — `lift_attackers`, `pass_back`, `pass_side` are near-zero in evolved agents; investigate and fix
2. **Island model + MAP-Elites** — diversity mechanisms

### Phase 4 groundwork (branch `ppo-gym-elo`, not yet merged)
Gymnasium env (`foosball_env.py`, `observation.py`), PPO hit policy (`ppo_strategy.py`, `train_ppo.py`), Elo ladder (`elo.py`). See `docs/ppo-notes.md`.

### Phase 3 — Game-Level Game Theory (first to 8)
- Iterated zero-sum game across points; each point is a round
- Opponent modeling via **Dirichlet belief state** over shot/defense tendencies, updated each point
- **Mixed Nash equilibrium** as target — stay unexploitable while exploiting opponent reads
- Score-state dependent policy (risk tolerance changes at 7-7 vs 7-2)

### Phase 4 — Reinforcement Learning
- Gymnasium wrapper around `simulate_point()`
- PPO via stable-baselines3, self-play curriculum
- Observation space: ball position, rod positions, score, point history, Dirichlet opponent belief state

### Phase 5 — Intra-Point Game Theory
- Extensive form game tree over decision nodes within a single point
- Fakes and misdirection — value comes from mixed equilibrium over real vs fake signals

### Phase 6 — Tournament Analysis
- Round-robin: evolved vs RL vs hand-coded vs game-theory agents
- TrueSkill ranking, strategy fingerprinting, co-evolution analysis

## Visualization

- Bird's-eye 2D field view with animated replay (`replay_point`)
- Ball position heatmap with gaussian blur overlay (`draw_stats`)
- Goal hit origin scatter coloured by scoring team (blue/red)
- Evolution multi-panel stats: fitness trajectory, gene composition over generations (`plot_evolution_stats`)
