"""Every strategy plays a few full points without crashing."""

import os

import numpy as np
import pytest

import config
from evolution import GENE_GROUPS, BASELINES, ParameterizedStrategy, load_agent
from field import BallState, Field
from simulation import simulate_point

OUTCOMES = {'goal', 'stallout', 'dead_ball', 'timeout'}
RANK0 = os.path.join(os.path.dirname(__file__), os.pardir, "output", "agents", "rank0.json")


@pytest.fixture(autouse=True)
def fps10(monkeypatch):
    """Pin 10 FPS explicitly: a full 6x6 sweep at 30 FPS would blow the fast-suite budget."""
    monkeypatch.setattr(config, "FPS", 10)
    monkeypatch.setattr(config, "DT", 0.1)
    monkeypatch.setattr(config, "MAX_TICKS", int(config.MAX_GAME_TIME * 10))


def check_result(r) -> None:
    assert r.outcome in OUTCOMES
    assert (r.winner is None) == (r.outcome in ('dead_ball', 'timeout'))
    if r.winner is not None:
        assert r.winner in (0, 1)
    assert 0 < r.ticks <= config.MAX_TICKS


def play(s0, s1, seed):
    return simulate_point(Field(), {0: s0, 1: s1}, kickoff_team=seed % 2,
                          seed=seed, record=True, collect_action_log=True)


@pytest.mark.parametrize("name0", BASELINES)
@pytest.mark.parametrize("name1", BASELINES)
def test_builtin_pairs(name0, name1):
    for seed in range(2):
        check_result(play(BASELINES[name0](), BASELINES[name1](), seed))


def random_genome(rng) -> np.ndarray:
    parts = [rng.uniform(0.0, 1.0, len(genes)) if name == "indep"
             else rng.dirichlet(np.ones(len(genes)))
             for name, genes in GENE_GROUPS.items()]
    return np.concatenate(parts)


def test_parameterized_no_controllable_rod(field):
    # 61d33b8: both hands mid-switch gives choose_pos an empty list
    strat = ParameterizedStrategy(random_genome(np.random.default_rng(0)))
    assert strat.choose_pos([], BallState(x=60.0, y=34.0), field) == {}


def test_parameterized_plays():
    rng = np.random.default_rng(1)
    for seed in range(4):
        s0 = ParameterizedStrategy(random_genome(rng))
        s1 = ParameterizedStrategy(random_genome(rng))
        check_result(play(s0, s1, seed))


@pytest.mark.skipif(not os.path.exists(RANK0), reason="no evolved agent at output/agents/rank0.json")
def test_rank0_agent_plays():
    agent, _ = load_agent(RANK0)
    for seed, name in enumerate(BASELINES):
        check_result(play(ParameterizedStrategy(agent.genome), BASELINES[name](), seed))
