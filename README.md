# Foosball Simulator

A time-stepped probabilistic foosball simulator where players, rods, and ball physics are modeled continuously with friction, reaction time, and hand-switching delays. Strategies control rod positioning and shot decisions each tick, and skill parameters add noise to shots. Built as a foundation for Monte Carlo analysis and eventually RL self-play agents.

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

# Simulate a single point, save a replay gif, and open it
python play.py
```

Edit the settings at the top of `play.py` to change the seed, strategies, skill levels, FPS, etc. Set `SHOW_LIVE = True` to watch in a matplotlib window instead of saving a gif.
