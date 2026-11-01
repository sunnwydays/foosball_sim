"""
foosball_env.py -- Gymnasium env: one episode = one point, one step = one tick.

The agent plays a random side against an opponent sampled from a pool each
reset: the 6 hardcoded baselines, the GA champion, and (once added) frozen PPO
snapshots. From the learner's side this is a single-agent task with
stochastic dynamics; self-play is just the pool growing.

Reward is +1 for scoring, -1 for conceding, 0 otherwise (draws and every tick
in between). No shaping. goal / stallout / dead_ball end the episode as
`terminated`; a 60 s timeout is `truncated` so PPO bootstraps V(s) instead of
treating the clock running out as a real ending.
"""

from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from evolution import BASELINES, ParameterizedStrategy
from field import Field
from observation import OBS_DIM, build_obs
from ppo_strategy import (ACTION_DIMS, GA_CHAMPION_PATH, PPOStrategy,
                          load_body_genome)
from simulation import iter_point


class FoosballEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, ga_path: str = GA_CHAMPION_PATH, p_snapshot: float = 0.5) -> None:
        self.observation_space = spaces.Box(-1.0, 1.0, (OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.MultiDiscrete(ACTION_DIMS)

        self.ga_genome = load_body_genome(ga_path)
        self.p_snapshot = p_snapshot
        self.snapshot_paths: list[str] = []
        self._snapshot_cache: dict[str, PPOStrategy] = {}
        self.field = Field()

        self.agent: Optional[PPOStrategy] = None
        self.opponent = None
        self.agent_team = 0
        self.opponent_name = ""
        self._gen = None
        self._view = None

    def add_opponent(self, path: str) -> None:
        """Add a frozen PPO snapshot to the opponent pool (called via env_method)."""
        self.snapshot_paths.append(path)

    def _sample_opponent(self):
        if self.snapshot_paths and self.np_random.random() < self.p_snapshot:
            path = self.snapshot_paths[int(self.np_random.integers(len(self.snapshot_paths)))]
            if path not in self._snapshot_cache:
                self._snapshot_cache[path] = PPOStrategy.from_file(path, self.ga_genome)
            return self._snapshot_cache[path], path
        names = list(BASELINES) + ["GA"]
        name = names[int(self.np_random.integers(len(names)))]
        if name == "GA":
            return ParameterizedStrategy(self.ga_genome), name
        return BASELINES[name](), name

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        # The sim draws from the global np.random; derive it from the env's own
        # stream so each worker process is reproducible and distinct.
        np.random.seed(int(self.np_random.integers(0, 2**31 - 1)))

        self.agent_team = int(self.np_random.integers(2))
        kickoff_team = int(self.np_random.integers(2))
        self.agent = PPOStrategy(self.ga_genome)
        self.opponent, self.opponent_name = self._sample_opponent()

        strategies = {self.agent_team: self.agent, 1 - self.agent_team: self.opponent}
        self._gen = iter_point(self.field, strategies, kickoff_team=kickoff_team)
        self._view = next(self._gen)
        return build_obs(self._view, self.agent_team), {}

    def step(self, action):
        self.agent.intent = (int(action[0]), int(action[1]), int(action[2]))
        if hasattr(self.opponent, "observe"):
            self.opponent.observe(self._view, 1 - self.agent_team)

        try:
            self._view = next(self._gen)
        except StopIteration as stop:
            result = stop.value
            if result.winner is None:
                reward = 0.0
            else:
                reward = 1.0 if result.winner == self.agent_team else -1.0
            info = {
                "outcome": result.outcome,
                "winner": result.winner,
                "is_self_goal": result.is_self_goal,
                "agent_team": self.agent_team,
                "opponent": self.opponent_name,
            }
            truncated = result.outcome == "timeout"
            # views are live references, so this is the final state
            return build_obs(self._view, self.agent_team), reward, not truncated, truncated, info

        return build_obs(self._view, self.agent_team), 0.0, False, False, {}
