# Foosball Simulator

A time-stepped probabilistic foosball simulator where players, rods, and ball physics are modeled continuously with friction, reaction time, and hand-switching delays. Strategies control rod positioning and shot decisions each tick, and skill parameters add noise to shots. Built as a foundation for Monte Carlo analysis and eventually RL self-play agents.

> [Progress slideshow](https://docs.google.com/presentation/d/1WwC7birlvtY4RJgm4wU8oJ-nC4ZbmzimYfUHUty_2lY)

## Contents

- [Setup](#setup)
- [Quick start](#quick-start)
- [How the simulation works](#how-the-simulation-works)
- [Project layout](#project-layout)
- [Tools](#tools)
- [Debugging workflow](#debugging-workflow)

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

## Quick start

| Command | What it does |
|---|---|
| `python play.py [seed]` | Simulate one point and open the scrubbable replay viewer |
| `python playground.py` | Physics sandbox: one rod, one ball, sliders for every contact parameter |
| `python main.py` | Run the Monte Carlo experiments (strategy matchups, 5000 points each) |
| `python evolution.py` | Evolve a parameterized strategy with a genetic algorithm |
| `python debug_scripts/find_seeds.py [n]` | Scan seeds `0..n-1` for known physics bugs and count them |
| `python debug_scripts/debug_face_slide.py <seed>` | Replay a face slide in isolation and trace every contact substep |
| `python debug_scripts/scan_stalls.py` | Find points that end by possession limit, time out, or stall |
| `python time_sim.py` | Benchmark simulation throughput (points per second) |

## How the simulation works

The table is 120 x 68 cm with 8 rods. Team 0 defends the left goal (x = 0) and attacks right; Team 1 defends the right goal (x = 120) and attacks left. Each team has two hands, so it can hold at most two rods at a time. The rest are **passive**.

A point is a loop of ticks (`dt = 1 / FPS`). Each tick, in [simulation.py](simulation.py):

1. Count down the reaction and hand-switch timers.
2. Each team's strategy picks which rods to hold, where to slide them, and whether to arm a swing. Passive rods are given a tilt and an up/down state.
3. Held rods slide toward their target y.
4. The ball moves in substeps of at most `CONTACT_STEP` cm, so fast shots can't skip a rod. [physics.py](physics.py) handles walls, goals, friction, and contact with **passive** players.
5. After each substep, **held** players that overlap the ball either connect with an armed swing (a hit) or rigid-bounce it (a whiff, or no swing armed).
6. Dead-ball and stall checks run, and the frame is recorded.

Rod rotation is modelled as an x-offset (`x_offset`, up to ±`ROD_X_REACH`). A rod can also be flipped **up** so the ball passes underneath.

## Project layout

### Engine

| File | Purpose |
|---|---|
| [config.py](config.py) | Every constant: field and rod geometry, speeds, friction, timing, skill noise, contact parameters. Units are cm, cm/s, and seconds. |
| [field.py](field.py) | Data structures: `Rod` (y-slide, x-offset tilt, up state, control state, armed swing), `Goal`, `BallState`, `TeamState`, `Field`. |
| [physics.py](physics.py) | `step_ball()`: ball motion, wall and goal checks, friction, and passive-player contact (pushback, reflect or glance, sideways deflection, flip-up). `ContactParams` holds the tunable contact constants. |
| [simulation.py](simulation.py) | `simulate_point()`: the tick loop above. Returns a `PointResult` with the winner, recorded `Frame`s, and the action log. |
| [strategy.py](strategy.py) | The `Strategy` interface (`choose_hands`, `choose_pos`, `choose_passive`, `choose_hit`) and the hand-coded strategies listed below. |

### Analysis and learning

| File | Purpose |
|---|---|
| [monte_carlo.py](monte_carlo.py) | `run_monte_carlo()`: plays many points and aggregates win rates, point lengths, and an optional ball-position heatmap. |
| [main.py](main.py) | A fixed set of strategy-vs-strategy and skill-gap experiments. Prints stats and saves heatmaps to `output/`. |
| [evolution.py](evolution.py) | Genetic algorithm over `ParameterizedStrategy` genomes: round-robin tournaments, crossover, mutation, and a benchmark against the hand-coded strategies. |
| [visualization.py](visualization.py) | Field drawing, GIF replays, the interactive replay viewer, stats and heatmap plots, and evolution plots. |

### Tools

| File | Purpose |
|---|---|
| [play.py](play.py) | Run and replay one point. The main way to look at the game. |
| [playground.py](playground.py) | Interactive single-rod physics sandbox. |
| [debug_scripts/find_seeds.py](debug_scripts/find_seeds.py) | Bug detector: scans seeds and flags suspicious ball-player contact. |
| [debug_scripts/debug_face_slide.py](debug_scripts/debug_face_slide.py) | Isolates and traces one flagged contact bug. |
| [debug_scripts/scan_stalls.py](debug_scripts/scan_stalls.py) | Lists seeds where a point stalls, times out, or ends by possession limit. |
| [debug_scripts/_path.py](debug_scripts/_path.py) | Puts the project root on `sys.path`; every debug script imports it first. |
| [time_sim.py](time_sim.py) | Measures throughput, single-core and in parallel. |

### Strategies

| Strategy | Behaviour |
|---|---|
| `SmackBall` | Tracks the ball's y and hits straight at the goal at full speed. |
| `AimAtGap` | Like `SmackBall`, but aims through the largest gap in the nearest opposing rod. |
| `HardOffense` | Shoots at a random point inside the goal at 1.5× normal hit speed. Lifts its passive offensive rods, and any held rod the ball has already passed, so they don't block its own shots. |
| `DefensiveWall` | Tilts passive defensive rods forward to block, flips passive offensive rods up, and shoots at goal center. |
| `TiltAndGap` | `DefensiveWall`'s passive positioning combined with gap-aimed shots. |
| `ReactiveBlock` | Tilts passive rods forward only when they are between the ball and its own goal, and lifts the ones behind the ball. |

Evolved agents are saved as `.json` genomes in `output/agents/` and can be used anywhere a strategy name is accepted, such as `TEAM_0 = "output/agents/rank0.json"` in `play.py`.

## Tools

### play.py: watch a point

```bash
python play.py        # uses SEED from the settings block
python play.py 136    # replay seed 136
```

Edit the settings block at the top of `play.py` to choose the strategies (a name or a genome `.json`), skills, anticipation, kickoff team, and FPS. To keep personal settings out of git, copy that block into `play_local.py`, which is gitignored and overrides it.

With `INTERACTIVE = True` it opens a scrubbable viewer:

- **Controls:** `space` play/pause, `←`/`→` move one second, `,`/`.` move one tick, `home`/`end`. There's also a frame slider and on-screen buttons.
- **Action log panel:** every hit, whiff, deflection, wall bounce, and goal, synced to playback. Click a line to jump to that event's frame.
- **Re-run panel:** change the seed, strategies, skills, or kickoff and re-simulate without closing the window.
- `SHOW_REACH` draws each player's hitbox. `SAVE_ACTION_LOG` also writes the log to `output/action_log_<timestamp>.txt`.

Set `INTERACTIVE = False` and `ANIMATE = True` to render `output/replay.gif` instead.

A seed fully determines a point. Given the same settings, `play.py <seed>` always replays the same game, and so does every tool below that imports `play.py`.

### playground.py: physics sandbox

Opens a mini field with one rod and sliders for ball velocity, friction, rod position and x-offset, and every passive contact parameter (slowdown, pushback, glance threshold, deflection, and so on). Click the field to place the ball and press Launch. Use it to get a feel for a contact constant before changing it in `config.py`.

### main.py: Monte Carlo experiments

Runs nine fixed matchups at 5000 points each: symmetric play, strategy against strategy, and a skill gap. It prints win rates and point statistics and saves heatmaps for two of them to `output/`.

### evolution.py: genetic algorithm

Evolves a population of `ParameterizedStrategy` genomes. Each generation:

1. Every pair plays a short round-robin (`RR_POINTS`).
2. Parents are selected by win rate.
3. Children are produced by crossover and mutation.
4. The generation's best is benchmarked against the hand-coded strategies.

Checkpoints go to `output/evolution_history.json` and `output/agents/best_latest.json` every generation. The top five agents are saved to `output/agents/rank{0..4}.json`. The final benchmark reports a z-score so a win is distinguished from noise. `python evolution.py --profile` runs a tiny population under `cProfile`.

### find_seeds.py: bug detector

```bash
python debug_scripts/find_seeds.py 300
```

Plays seeds `0..n-1` with `play.py`'s settings and prints every suspicious contact, then a totals line:

- **BACK-BOUNCE:** a held rod rigid-bounces a ball that was moving toward the opponent's goal, sending it back toward its own goal (the player was hit from behind).
- **FACE-SLIDE:** the ball stays on a passive player's face for several ticks in a row and the owning team concedes within a second.

Every printed seed replays exactly with `python play.py <seed>`. The totals line is the before/after number for any physics fix.

### debug_face_slide.py: isolate and trace

```bash
python debug_scripts/debug_face_slide.py 136                  # first face slide in seed 136
python debug_scripts/debug_face_slide.py 136 --tick 26        # start from a chosen tick
python debug_scripts/debug_face_slide.py --manual --xo -3 --dy 1 --vx 150
```

For a seed, it:

1. **Captures:** runs the point, finds the slide with `find_seeds`' detector, and prints the recorded frames around it.
2. **Isolates:** copies the ball and the sliding rod into a fresh `Field`, with every other rod lifted.
3. **Replays twice:** once with physics only, and once with the strategy's `choose_passive()` reapplied every tick, as in the real game. If only the second run fails, the bug involves the strategy, not just the physics.
4. **Traces:** prints one line per contact substep (vx/vy before and after, rod offset, `offset_toward_ball`, and whether the contact reflected or glanced), then a PASS/FAIL verdict.

It wraps `physics._step_ball_inner` at runtime, so edits to `physics.py` show up in the trace immediately. `--manual` builds a shot at the T1 goalie with a chosen tilt, offset, and speed instead of replaying a seed.

### scan_stalls.py and time_sim.py

`debug_scripts/scan_stalls.py` plays 300 seeds of HardOffense vs TiltAndGap. It prints points that ended by possession limit, timed out, or ran over 6 s with few hits. `time_sim.py` reports points per second for several matchups on one core and for a parallel round-robin generation. Run it before and after performance changes.

## Debugging workflow

The face-slide fix (commit `270d45b`) followed these steps.

1. **Count it.** Add a detector to `find_seeds.py` and run it for a seed list and a baseline count.
2. **Watch it.** Replay a flagged seed with `play.py <seed>` and find the pattern in the action log.
3. **Isolate it.** Replay the state just before the bug with everything else removed (`debug_face_slide.py`, `playground.py`).
4. **Trace it.** Print every contact substep, since per-tick output hides what happens.
5. **Hypothesize.** Decide what should physically happen, and check it against the trace before you edit anything.
6. **Change one thing.** Fix the cause rather than tuning a constant to hide it.
7. **Verify.** Re-run the trace, then `find_seeds.py`. The bug count should drop and the other counts should not rise.
8. **Commit.** Include the cause and the before/after counts in the message.
