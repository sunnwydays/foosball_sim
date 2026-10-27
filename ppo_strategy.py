"""
ppo_strategy.py -- a learned policy driving the GA's strategy body.

PPOStrategy = ParameterizedStrategy (hands, positioning, passive rods, skills
all from a fixed GA genome) with choose_hit replaced by a policy over three
discrete choices per tick:

  aim    0..5  gap / off wall / off player / pass fwd / pass back / pass side
  speed  0..2  shot speed = HIT_SPEED * (0.5 + 0.5 * s),  s in {0, 0.5, 1}
  commit 0..1  0 = hold, 1 = ask _commit_swing to arm a swing (it still owns timing)

Because the body is the GA champion's, any rating gap between this agent and
the GA champion comes from the hit policy alone.

The policy acts once per tick, not once per rod: the same action applies to
every free rod. Where the action comes from:
  * training: the env sets `intent` before each tick
  * frozen opponent / ladder: `observe(view, team)` runs the saved model
"""

import os
from typing import Optional

import numpy as np

import config
from evolution import GENOME_SIZE, ParameterizedStrategy, load_agent
from field import BallState, Field, Rod, SwingCommitment
from observation import build_obs
from simulation import PointResult, TickView, iter_point

GA_CHAMPION_PATH = "output/agents/rank0_latest.json"
ACTION_DIMS = [6, 3, 2]   # aim, speed, commit


def neutral_genome() -> np.ndarray:
    """Every group uniform, every independent gene 0.5. Fallback when no GA file exists."""
    g = np.full(GENOME_SIZE, 0.5)
    g[:9] = 1.0 / 3.0   # skill, shot, pass groups each sum to 1
    return g


def load_body_genome(path: str = GA_CHAMPION_PATH) -> np.ndarray:
    if os.path.exists(path):
        return load_agent(path)[0].genome
    return neutral_genome()


class PPOStrategy(ParameterizedStrategy):
    def __init__(self, genome: Optional[np.ndarray] = None, model=None) -> None:
        super().__init__(load_body_genome() if genome is None else genome)
        self.model = model                       # SB3 PPO, or None while training
        self.intent: Optional[tuple[int, int, int]] = None

    @classmethod
    def from_file(cls, path: str, genome: Optional[np.ndarray] = None) -> "PPOStrategy":
        import torch
        from stable_baselines3 import PPO
        torch.set_num_threads(1)   # many worker processes: don't oversubscribe cores
        return cls(genome, PPO.load(path, device="cpu"))

    def observe(self, view: TickView, team: int) -> None:
        """Pick this tick's action from the model (no-op while training)."""
        if self.model is not None:
            action, _ = self.model.predict(build_obs(view, team), deterministic=False)
            self.intent = (int(action[0]), int(action[1]), int(action[2]))

    def choose_hit(
        self,
        rod: Rod,
        ball: BallState,
        field: Field,
        t: float,
    ) -> Optional[SwingCommitment]:
        rod.accuracy          = self.accuracy
        rod.power_consistency = self.power_consistency
        rod.anticipation      = self.anticipation

        if self.intent is None:
            return None
        aim, speed, commit = self.intent
        if commit == 0:
            return None

        aim_x, aim_y = self._aim_target(aim, rod, ball, field)
        intended_speed = config.HIT_SPEED * (0.5 + 0.25 * speed)
        return self._commit_swing(rod, ball, field, t, aim_x, aim_y, intended_speed)


def play_point(field: Field, strategies: dict, kickoff_team: int = 0,
               seed: Optional[int] = None) -> PointResult:
    """simulate_point that also lets model-driven strategies observe each tick."""
    gen = iter_point(field, strategies, kickoff_team=kickoff_team, seed=seed)
    try:
        view = next(gen)
        while True:
            for team, strat in strategies.items():
                if hasattr(strat, "observe"):
                    strat.observe(view, team)
            view = next(gen)
    except StopIteration as stop:
        return stop.value
