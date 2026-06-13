"""
monte_carlo.py — Monte Carlo runner and result aggregation.

Usage
-----
    from monte_carlo import run_monte_carlo
    from field import Field
    from strategy import SmackBall

    field      = Field()
    strategies = {0: SmackBall(), 1: SmackBall()}
    result     = run_monte_carlo(field, strategies, n_simulations=1_000)
    print(result)
"""

from __future__ import annotations

from dataclasses import dataclass
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
    n_simulations      : total runs.
    team0_wins         : count of points won by Team 0.
    team1_wins         : count of points won by Team 1.
    draws              : time-limit / no-winner points.
    team0_win_rate     : team0_wins / n_simulations.
    team1_win_rate     : team1_wins / n_simulations.
    draw_rate          : draws / n_simulations.
    avg_ticks          : mean ticks per point.
    avg_time           : mean game-time per point (seconds).
    pos_grid           : normalized ball position heatmap (if track_stats=True).
    goal_hits          : (x, y) of last active hit before each goal (if track_stats=True).
    """
    n_simulations:    int
    team0_wins:       int
    team1_wins:       int
    draws:            int
    team0_win_rate:   float
    team1_win_rate:   float
    draw_rate:        float
    avg_ticks:        float
    avg_time:         float
    pos_grid:         Optional[np.ndarray] = None
    goal_hits:        Optional[list]       = None
    stallouts:        int                  = 0
    dead_ball_draws:  float                = 0.0
    timeout_draws:    float                = 0.0
    team0_self_goals: float                = 0.0
    team1_self_goals: float                = 0.0
    stall_hits:       Optional[list]       = None
    dead_hits:        Optional[list]       = None

    def __str__(self) -> str:
        bar0 = "#" * round(self.team0_win_rate * 40)
        bar1 = "#" * round(self.team1_win_rate * 40)
        n = self.n_simulations
        return (
            f"--- Monte Carlo Results ({n:,} simulations) ---\n"
            f"  Team 0  {self.team0_win_rate:>6.1%}  {bar0}\n"
            f"  Team 1  {self.team1_win_rate:>6.1%}  {bar1}\n"
            f"  Draws   {self.draw_rate:>6.1%}"
            f"  (dead-ball {self.dead_ball_draws:>5.1%}  timeout {self.timeout_draws:>5.1%})\n"
            f"  Stallouts       {self.stallouts:>5,}  ({self.stallouts/n:>5.1%})\n"
            f"  Own goals   T0:{self.team0_self_goals:>5.1%}  T1:{self.team1_self_goals:>5.1%}\n"
            f"  Avg ticks/point : {self.avg_ticks:.1f} ({self.avg_time:.2f}s)\n"
        )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_monte_carlo(
    field:          Field,
    strategies:     dict[int, Strategy],
    n_simulations:  int   = 1_000,
    p_team0_starts: float = 0.5,
    seed:           Optional[int] = None,
    track_stats:    bool  = False,
) -> MCResult:
    """
    Run `n_simulations` independent point simulations and aggregate results.

    Parameters
    ----------
    field              : Field instance (reset before each sim).
    strategies         : {team_id: Strategy}.
    n_simulations      : number of independent points to simulate.
    p_team0_starts     : probability that Team 0 kicks off each point.
    seed               : optional RNG seed for reproducibility.

    Returns
    -------
    MCResult with win rates and timing statistics.
    """
    if seed is not None:
        np.random.seed(seed)

    wins       = [0, 0]
    draws      = 0
    all_ticks: list[int]   = []
    all_times: list[float] = []

    stallouts        = 0
    dead_ball_draws  = 0
    timeout_draws    = 0
    self_goals       = [0, 0]

    # Stats tracking
    depth_cells = int(field.depth / config.STATS_GRID_RES)
    width_cells = int(field.width / config.STATS_GRID_RES)
    pos_grid   = np.zeros((depth_cells, width_cells)) if track_stats else None
    goal_hits:  list = [] if track_stats else None
    stall_hits: list = [] if track_stats else None
    dead_hits:  list = [] if track_stats else None

    for _ in range(n_simulations):
        kickoff_team = 0 if np.random.random() < p_team0_starts else 1

        result: PointResult = simulate_point(
            field, strategies,
            kickoff_team=kickoff_team,
            record=False,
            pos_grid=pos_grid,
            goal_hits=goal_hits,
            stall_hits=stall_hits,
            dead_hits=dead_hits,
        )

        all_ticks.append(result.ticks)
        all_times.append(result.time)

        if result.outcome == 'stallout':
            stallouts += 1
            wins[result.winner] += 1
        elif result.outcome == 'dead_ball':
            dead_ball_draws += 1
            draws += 1
        elif result.outcome == 'timeout':
            timeout_draws += 1
            draws += 1
        else:  # 'goal'
            wins[result.winner] += 1
            if result.is_self_goal:
                self_goals[1 - result.winner] += 1  # team that shot into own net

    # Normalize grid to [0, 1]
    if pos_grid is not None and pos_grid.max() > 0:
        pos_grid = pos_grid / pos_grid.max()

    total = n_simulations
    return MCResult(
        n_simulations    = total,
        team0_wins       = wins[0],
        team1_wins       = wins[1],
        draws            = draws,
        team0_win_rate   = wins[0] / total,
        team1_win_rate   = wins[1] / total,
        draw_rate        = draws / total,
        avg_ticks        = float(np.mean(all_ticks)),
        avg_time         = float(np.mean(all_times)),
        pos_grid         = pos_grid,
        goal_hits        = goal_hits,
        stallouts        = stallouts,
        dead_ball_draws  = dead_ball_draws / total,
        timeout_draws    = timeout_draws / total,
        team0_self_goals = self_goals[0] / total,
        team1_self_goals = self_goals[1] / total,
        stall_hits       = stall_hits,
        dead_hits        = dead_hits,
    )
