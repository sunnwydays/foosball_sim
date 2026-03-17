"""
monte_carlo.py — Monte Carlo runner and result aggregation.

What is Monte Carlo simulation?
--------------------------------
Run the same probabilistic scenario N times (each run is independent),
collect the outcomes, and compute statistics over that distribution of results.

Here, one "run" = one foosball point (simulate_point). Because each point
starts fresh with random noise on every shot, outcomes vary. Running 10,000
points gives us a reliable estimate of:

  - Win probability for each team
  - Average number of possession changes per point
  - Distribution of outcomes (for histograms, heatmaps, etc.)

The more simulations, the lower the estimation error.
Rule of thumb: 10,000 sims gives ~±1% accuracy on win rates.

Usage
-----
    from monte_carlo import run_monte_carlo
    from field import Field
    from strategy import AimAtGoalCenter

    field    = Field()
    strategies = {0: AimAtGoalCenter(), 1: AimAtGoalCenter()}
    result   = run_monte_carlo(field, strategies, n_simulations=10_000)
    print(result)
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Optional

import numpy as np

import config
from field import Field
from strategy import Strategy
from simulation import simulate_point, PointResult


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class MCResult:
    """
    Aggregated results from N Monte Carlo point simulations.

    Attributes
    ----------
    n_simulations         : total runs.
    team0_wins            : count of points won by Team 0.
    team1_wins            : count of points won by Team 1.
    draws                 : dead balls / turn-limit points (no winner).
    team0_win_rate        : team0_wins / n_simulations.
    team1_win_rate        : team1_wins / n_simulations.
    draw_rate             : draws / n_simulations.
    avg_turns             : mean possession changes across all points.
    avg_turns_decided     : mean possession changes for points with a winner.
    all_trajectories      : list of trajectories (only stored if requested).
    """
    n_simulations:      int
    team0_wins:         int
    team1_wins:         int
    draws:              int
    team0_win_rate:     float
    team1_win_rate:     float
    draw_rate:          float
    avg_turns:          float
    avg_turns_decided:  float
    all_trajectories:   list = dc_field(default_factory=list, repr=False)

    def __str__(self) -> str:
        bar0 = "#" * round(self.team0_win_rate * 40)
        bar1 = "#" * round(self.team1_win_rate * 40)
        return (
            f"--- Monte Carlo Results ({self.n_simulations:,} simulations) ---\n"
            f"  Team 0  {self.team0_win_rate:>6.1%}  {bar0}\n"
            f"  Team 1  {self.team1_win_rate:>6.1%}  {bar1}\n"
            f"  Draws   {self.draw_rate:>6.1%}\n"
            f"  Avg turns/point          : {self.avg_turns:.1f}\n"
            f"  Avg turns (decided only) : {self.avg_turns_decided:.1f}\n"
        )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_monte_carlo(
    field:          Field,
    strategies:     dict[int, Strategy],
    n_simulations:  int   = 10_000,
    p_team0_starts: float = 0.5,
    seed:           Optional[int] = None,
    store_trajectories: bool = False,
) -> MCResult:
    """
    Run `n_simulations` independent point simulations and aggregate results.

    Parameters
    ----------
    field               : Field instance (rod offsets are reset before each sim).
    strategies          : {team_id: Strategy} — one strategy object per team.
    n_simulations       : number of independent points to simulate.
    p_team0_starts      : probability that Team 0 kicks off each point.
                          0.5 = fair coin flip (default); 1.0 = Team 0 always;
                          0.0 = Team 1 always.
    seed                : optional RNG seed for reproducibility.
    store_trajectories  : if True, attach all trajectories to MCResult
                          (memory-intensive for large N — use for debugging).

    Returns
    -------
    MCResult with win rates and turn statistics.
    """
    if seed is not None:
        np.random.seed(seed)

    wins           = [0, 0]
    draws          = 0
    all_turns:     list[int]   = []
    decided_turns: list[int]   = []
    trajectories:  list        = []

    for _ in range(n_simulations):
        # Randomly assign kickoff each point based on p_team0_starts
        starting_team   = 0 if np.random.random() < p_team0_starts else 1
        kickoff_rod_idx = config.KICKOFF_ROD[starting_team]

        # Reset rod positions to defaults before every point
        field.reset_rod_offsets()

        result: PointResult = simulate_point(field, strategies, kickoff_rod_idx)

        all_turns.append(result.turns)

        if result.winner is not None:
            wins[result.winner] += 1
            decided_turns.append(result.turns)
        else:
            draws += 1

        if store_trajectories:
            trajectories.append(result.trajectory)

    total = n_simulations
    avg_decided = float(np.mean(decided_turns)) if decided_turns else 0.0

    return MCResult(
        n_simulations     = total,
        team0_wins        = wins[0],
        team1_wins        = wins[1],
        draws             = draws,
        team0_win_rate    = wins[0] / total,
        team1_win_rate    = wins[1] / total,
        draw_rate         = draws / total,
        avg_turns         = float(np.mean(all_turns)),
        avg_turns_decided = avg_decided,
        all_trajectories  = trajectories,
    )
