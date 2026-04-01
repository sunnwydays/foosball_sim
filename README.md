# Foosball Simulator

A time-stepped probabilistic foosball simulator where players, rods, and ball physics are modeled continuously with friction, reaction time, and hand-switching delays. Strategies control rod positioning and shot decisions each tick, and skill parameters add noise to shots. Built as a foundation for Monte Carlo analysis and eventually RL self-play agents.

> [Progress slideshow](https://docs.google.com/presentation/d/1WwC7birlvtY4RJgm4wU8oJ-nC4ZbmzimYfUHUty_2lY)

## Setup

```bash
python -m venv venv
venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

## Running

```bash
# Run all experiments (Monte Carlo stats printed to console)
python main.py

# Simulate and visualize a single point
python play.py

# Interactive physics playground — tune contact params with sliders
python playground.py
```

Edit the settings at the top of [play.py](play.py) to change the seed, strategies, skill levels, and FPS. The replay gif is saved to `output/replay.gif` and opened automatically in your default viewer. Set `SHOW_LIVE = True` to watch it in a live matplotlib window instead.

### Playground

`python playground.py` opens an interactive matplotlib window with a mini field, one rod, and sliders for ball velocity, friction, rod position, and all passive contact parameters (slowdown, pushback, glance threshold, deflection, etc.). Click the field to place the ball, then press Launch to watch it interact with the rod. Useful for tuning physics constants before running full simulations.
