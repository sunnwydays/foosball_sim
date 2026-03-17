"""
main.py -Entry point and demonstration experiments.

Run with:  python main.py

Three experiments are run back-to-back:

  1. Symmetric baseline -both teams use AimAtGoalCenter at equal skill.
     Win rates should be ~50/50 (any deviation is symmetry-breaking from
     which team kicks off first).

  2. Strategy comparison -AimAtGoalCenter vs AimAtRandomGoalPoint.
     Tests whether predictable aim (always goal-center) is exploitable by
     a defender who also tracks the ball.

  3. Skill gap -high-skill Team 0 vs low-skill Team 1, same strategy.
     Demonstrates how the skill parameter (shot tightness) affects win rate.

Outputs
-------
PNG files are saved to ./output/ -one field layout and one heatmap per experiment.
"""

import matplotlib
matplotlib.use("Agg")   # non-interactive backend -works without a display
import matplotlib.pyplot as plt
import os

from field import Field
from strategy import AimAtGoalCenter, AimAtRandomGoalPoint, RandomStrategy
from monte_carlo import run_monte_carlo
from visualization import draw_field, draw_heatmap

N = 10_000   # simulations per experiment (increase for more accuracy)
OUTPUT_DIR = "output"


def _save(filename: str) -> None:
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close("all")
    print(f"  -> saved {path}")


def run_experiment(
    title: str,
    field: Field,
    strategies: dict,
    seed: int = 42,
    p_team0_starts: float = 0.5,
    slug: str = "experiment",
) -> None:
    print(f"\n{'=' * 55}")
    print(f"  {title}")
    print(f"{'=' * 55}")
    result = run_monte_carlo(
        field, strategies, n_simulations=N, seed=seed,
        p_team0_starts=p_team0_starts, store_trajectories=True,
    )
    print(result)

    # Field layout (static snapshot -no ball)
    draw_field(field, title=title)
    _save(f"{slug}_field.png")

    # Possession heatmap across all simulated points
    draw_heatmap(field, result.all_trajectories, title=f"{title} -Heatmap")
    _save(f"{slug}_heatmap.png")


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(Field().describe())   # show the table layout once

    # ------------------------------------------------------------------
    # Experiment 1: Symmetric baseline
    # ------------------------------------------------------------------
    run_experiment(
        "Experiment 1 - Symmetric: both AimAtGoalCenter, equal skill",
        field      = Field(),
        strategies = {0: AimAtGoalCenter(), 1: AimAtGoalCenter()},
        slug       = "exp1_symmetric",
    )

    # ------------------------------------------------------------------
    # Experiment 2: Strategy comparison
    # ------------------------------------------------------------------
    run_experiment(
        "Experiment 2 - AimAtGoalCenter (T0) vs AimAtRandomGoalPoint (T1)",
        field      = Field(),
        strategies = {0: AimAtGoalCenter(), 1: AimAtRandomGoalPoint()},
        slug       = "exp2_strategy",
    )

    # ------------------------------------------------------------------
    # Experiment 3: Skill gap
    # ------------------------------------------------------------------
    skilled_field = Field()
    for rod in skilled_field.rods:
        if rod.team == 0:
            rod.skill       = 0.9   # tight angle distribution
            rod.consistency = 0.9   # consistent speed
        else:
            rod.skill       = 0.2   # wide, noisy shots
            rod.consistency = 0.2

    run_experiment(
        "Experiment 3 - Skill gap: Team 0 (skill=0.9) vs Team 1 (skill=0.2)",
        field      = skilled_field,
        strategies = {0: AimAtGoalCenter(), 1: AimAtGoalCenter()},
        slug       = "exp3_skill_gap",
    )

    # ------------------------------------------------------------------
    # Experiment 4: Random baseline
    # ------------------------------------------------------------------
    run_experiment(
        "Experiment 4 - AimAtGoalCenter (T0) vs Random (T1)",
        field      = Field(),
        strategies = {0: AimAtGoalCenter(), 1: RandomStrategy()},
        slug       = "exp4_random",
    )



if __name__ == "__main__":
    main()
