"""
main.py - Entry point and demonstration experiments.

Run with:  python main.py

Experiments use the time-stepped continuous simulation engine.
"""

import matplotlib.pyplot as plt

from field import Field
from strategy import (SmackBall, AimAtGap, HardOffense,
                      DefensiveWall, TiltAndGap, ReactiveBlock)
from monte_carlo import run_monte_carlo
from visualization import draw_stats


N = 5000   # simulations per experiment


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

    if result5.pos_grid is not None:
        _, ax = plt.subplots(figsize=(14, 7), facecolor="#1a1a1a")
        draw_stats(Field(), result5.pos_grid, result5.goal_hits or [],
                   title="Exp 5 — HardOffense vs AimAtGap", ax=ax)
        plt.show()

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

    if result9.pos_grid is not None:
        _, ax = plt.subplots(figsize=(14, 7), facecolor="#1a1a1a")
        draw_stats(Field(), result9.pos_grid, result9.goal_hits or [],
                   title="Exp 9 — ReactiveBlock vs DefensiveWall", ax=ax)
        plt.show()

if __name__ == "__main__":
    main()
