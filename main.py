"""
main.py - Entry point and demonstration experiments.

Run with:  python main.py

Experiments use the time-stepped continuous simulation engine.
"""

from field import Field
from strategy import SmackBall, AimAtGap
from monte_carlo import run_monte_carlo


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

    # Experiment 1: Symmetric baseline
    run_experiment(
        "Exp 1 - Symmetric: both SmackBall, equal skill",
        field      = Field(),
        strategies = {0: SmackBall(), 1: SmackBall()},
    )

    # Experiment 2: Strategy comparison
    run_experiment(
        "Exp 2 - SmackBall (T0) vs AimAtGap (T1)",
        field      = Field(),
        strategies = {0: SmackBall(), 1: AimAtGap()},
    )

    # Experiment 3: Skill gap
    skilled_field = Field()
    for rod in skilled_field.rods:
        if rod.team == 0:
            rod.skill, rod.consistency = 0.9, 0.9
        else:
            rod.skill, rod.consistency = 0.2, 0.2

    run_experiment(
        "Exp 3 - Skill gap: T0 (0.9) vs T1 (0.2), both SmackBall",
        field      = skilled_field,
        strategies = {0: SmackBall(), 1: SmackBall()},
    )


if __name__ == "__main__":
    main()
