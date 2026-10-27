"""Bradley-Terry fit recovers known Elo gaps."""

import numpy as np

from elo import fit_bt


def test_fit_recovers_known_gaps():
    # true Elo 1600 / 1500 / 1400: P(a beats b) = 1 / (1 + 10 ** (-(ea - eb) / 400))
    elo = {"a": 1600.0, "b": 1500.0, "c": 1400.0}
    n = 100_000
    results = {}
    for x, y in (("a", "b"), ("a", "c"), ("b", "c")):
        p = 1 / (1 + 10 ** (-(elo[x] - elo[y]) / 400))
        results[f"{x}|{y}"] = [round(p * n), n - round(p * n), 0]
    fit = fit_bt(["a", "b", "c"], results)
    np.testing.assert_allclose(fit, [1600, 1500, 1400], atol=2)


def test_draws_count_half():
    fit = fit_bt(["a", "b"], {"a|b": [0, 0, 50]})
    np.testing.assert_allclose(fit, [1500, 1500], atol=1e-6)
