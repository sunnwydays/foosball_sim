"""Gymnasium env, observation mirroring, PPOStrategy action mapping."""

import numpy as np
import pytest

import config
from field import BallState, Field, TeamState
from foosball_env import FoosballEnv
from observation import OBS_DIM, build_obs
from ppo_strategy import PPOStrategy, neutral_genome
from simulation import TickView
from tests.helpers import hold


def _view(field, ball, timers=(1.0, 2.0)):
    return TickView(ball, field, {0: TeamState(), 1: TeamState()}, dict(enumerate(timers)), 0.0)


def test_check_env():
    from stable_baselines3.common.env_checker import check_env
    check_env(FoosballEnv(), skip_render_check=True)


def test_random_episodes_end_with_valid_rewards():
    env = FoosballEnv()
    for seed in range(5):
        obs, _ = env.reset(seed=seed)
        for _ in range(config.MAX_TICKS + 5):
            assert obs.shape == (OBS_DIM,) and np.all(np.abs(obs) <= 1.0)
            obs, reward, term, trunc, info = env.step(env.action_space.sample())
            assert reward in (-1.0, 0.0, 1.0)
            if term or trunc:
                assert info["outcome"] in ("goal", "stallout", "dead_ball", "timeout")
                assert (reward != 0.0) == (info["winner"] is not None)
                break
        else:
            pytest.fail("episode never ended")


def test_mirrored_state_gives_same_obs():
    """Team 0's view of a state equals team 1's view of its x-mirror."""
    f0, f1 = Field(), Field()
    ball0 = BallState(x=30.0, y=20.0, vx=50.0, vy=-10.0)
    ball1 = BallState(x=f0.depth - 30.0, y=20.0, vx=-50.0, vy=-10.0)

    rng = np.random.default_rng(0)
    by_x = lambda f, team: sorted((r for r in f.rods if r.team == team), key=lambda r: r._base_x)
    for r0, r1 in zip(by_x(f0, 0), reversed(by_x(f1, 1))):   # team 0 rod <-> mirrored team 1 rod
        y, x = rng.uniform(-5, 5), rng.uniform(-3, 3)
        r0.y_offset, r0.x_offset, r0.up = y, x, bool(rng.integers(2))
        r1.y_offset, r1.x_offset, r1.up = y, -x, r0.up
        hold(r0)
        hold(r1)
    for r0, r1 in zip(by_x(f0, 1), reversed(by_x(f1, 0))):   # opponents likewise
        y, x = rng.uniform(-5, 5), rng.uniform(-3, 3)
        r0.y_offset, r0.x_offset = y, x
        r1.y_offset, r1.x_offset = y, -x

    np.testing.assert_allclose(build_obs(_view(f0, ball0), 0), build_obs(_view(f1, ball1, (2.0, 1.0)), 1), atol=1e-6)


def test_ppo_hold_returns_none_and_commit_arms_swing(field):
    strat = PPOStrategy(neutral_genome())
    strat.anticipation = 1.0                       # no TTC noise: the commit decision is deterministic
    rod = hold(field.rods[3])                      # T0 mid
    ball = BallState(x=rod.x - 16.0, y=34.0, vx=40.0, vy=0.0)  # ttc 0.35 s: inside the 0.3-0.4 s lead band

    strat.intent = (0, 1, 0)
    assert strat.choose_hit(rod, ball, field, 0.0) is None

    strat.intent = (0, 2, 1)
    swing = strat.choose_hit(rod, ball, field, 0.0)
    assert swing is not None and swing.vx > 0
