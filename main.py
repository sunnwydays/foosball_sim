"""
main.py - Entry point and demonstration experiments.

Run with:  python main.py

Experiments use the time-stepped continuous simulation engine.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from field import Field
from strategy import (SmackBall, AimAtGap, HardOffense,
                      DefensiveWall, TiltAndGap, ReactiveBlock)
from monte_carlo import run_monte_carlo
from visualization import draw_stats


N = 5000   # simulations per experiment


def save_heatmap(field: Field, result, title: str, filename: str) -> None:
    """Render a heatmap to output/<filename> and open it (non-blocking)."""
    if result.pos_grid is None:
        return
    _, ax = plt.subplots(figsize=(14, 7), facecolor="#1a1a1a")
    draw_stats(field, result.pos_grid, result.goal_hits or [],
               title=title, ax=ax)
    os.makedirs("output", exist_ok=True)
    path = os.path.join("output", filename)
    plt.savefig(path, facecolor="#1a1a1a", dpi=120)
    plt.close()
    print(f"Saved heatmap to {path}")
    os.startfile(os.path.abspath(path))


def run_experiment(
    title: str,
    field: Field,
    strategies: dict,
    seed: int = 42,
    p_team0_starts: float = 0.5,
) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")
    result = run_monte_carlo(
        field, strategies, n_simulations=N, seed=seed,
        p_team0_starts=p_team0_starts,
    )
    print(result)


def main() -> None:
    print(Field().describe())

    run_experiment(
        "Exp 1 - Symmetric: both SmackBall, equal skill",
        field      = Field(),
        strategies = {0: SmackBall(), 1: SmackBall()},
    )

    run_experiment(
        "Exp 2 - SmackBall vs AimAtGap",
        field      = Field(),
        strategies = {0: SmackBall(), 1: AimAtGap()},
    )

    skilled_field = Field()
    for rod in skilled_field.rods:
        if rod.team == 0:
            rod.accuracy, rod.power_consistency = 0.9, 0.9
        else:
            rod.accuracy, rod.power_consistency = 0.2, 0.2

    run_experiment(
        "Exp 3 - Skill gap: Smackball (0.9) vs (0.2)",
        field      = skilled_field,
        strategies = {0: SmackBall(), 1: SmackBall()},
    )

    run_experiment(
        "Exp 4 - SmackBall vs HardOffense",
        field      = Field(),
        strategies = {0: SmackBall(), 1: HardOffense()},
    )

    result5 = run_monte_carlo(
        Field(), {0: HardOffense(), 1: AimAtGap()},
        n_simulations=N, seed=42,
        track_stats=True,
    )
    print(f"\n{'=' * 60}")
    print(f"  Exp 5 - HardOffense vs AimAtGap")
    print(f"{'=' * 60}")
    print(result5)

    save_heatmap(Field(), result5,
                 title="Exp 5 — HardOffense vs AimAtGap",
                 filename="exp5_hardoffense_vs_aimatgap.png")

    run_experiment(
        "Exp 6 - DefensiveWall vs HardOffense",
        field      = Field(),
        strategies = {0: DefensiveWall(), 1: HardOffense()},
    )

    run_experiment(
        "Exp 7 - TiltAndGap vs SmackBall",
        field      = Field(),
        strategies = {0: TiltAndGap(), 1: SmackBall()},
    )

    run_experiment(
        "Exp 8 - ReactiveBlock vs TiltAndGap",
        field      = Field(),
        strategies = {0: ReactiveBlock(), 1: TiltAndGap()},
    )

    result9 = run_monte_carlo(
        Field(), {0: ReactiveBlock(), 1: DefensiveWall()},
        n_simulations=N, seed=42,
        track_stats=True,
    )
    print(f"\n{'=' * 60}")
    print(f"  Exp 9 - ReactiveBlock vs DefensiveWall")
    print(f"{'=' * 60}")
    print(result9)

    save_heatmap(Field(), result9,
                 title="Exp 9 — ReactiveBlock vs DefensiveWall",
                 filename="exp9_reactiveblock_vs_defensivewall.png")

if __name__ == "__main__":
    main()
