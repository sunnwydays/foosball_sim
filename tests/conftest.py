"""
Shared fixtures. Importable helpers (ScriptedStrategy, ball_at, ...) live in
tests/helpers.py so test files can import them by name.

Never import play.py from a test: on import it rewrites config.FPS/DT/MAX_TICKS,
pulls in the gitignored play_local.py overrides and loads matplotlib, so the
suite would depend on whoever's machine it runs on.
"""

import pytest

import config
from field import Field


@pytest.fixture(params=[10, 30], ids=lambda f: f"{f}fps")
def fps(request, monkeypatch):
    """Run the test at 10 and 30 FPS (substep tunnelling only showed at 10).

    Safe to monkeypatch: every module reads config.DT / MAX_TICKS at call
    time, none copies them on import.
    """
    monkeypatch.setattr(config, "FPS", request.param)
    monkeypatch.setattr(config, "DT", 1.0 / request.param)
    monkeypatch.setattr(config, "MAX_TICKS", int(config.MAX_GAME_TIME * request.param))
    return request.param


@pytest.fixture
def field():
    return Field()
