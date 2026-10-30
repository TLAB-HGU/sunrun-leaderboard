import numpy as np

from context import asymmetric_loss, context_objective, event_labels, purged_blocks


def test_gradient_and_hessian():
    r = np.random.default_rng(42)
    y = r.normal(450, 30, (5, 24)); p = r.normal(440, 30, y.shape); q = r.random(5)
    g, h = context_objective(q, 1)(y.ravel(), p)
    delta = 1e-4
    numeric = (asymmetric_loss(y, p + delta, q, 1) - asymmetric_loss(y, p - delta, q, 1)) / (2 * delta)
    np.testing.assert_allclose(g, numeric, rtol=1e-6, atol=1e-6)
    gp = context_objective(q, 1)(y.ravel(), p + delta)[0]
    gm = context_objective(q, 1)(y.ravel(), p - delta)[0]
    np.testing.assert_allclose(h, (gp - gm) / (2 * delta), rtol=1e-6)
    assert np.all(h >= 1)


def test_zero_strength_and_gate():
    y = np.ones((2, 24)) * 500; p = y - 25
    for q, lam in [(np.ones(2), 0), (np.zeros(2), 1)]:
        g, h = context_objective(q, lam)(y.ravel(), p)
        np.testing.assert_array_equal(g, p - y)
        np.testing.assert_array_equal(h, np.ones_like(p))


def test_labels_and_purge():
    y = np.ones((3, 24)) * 300
    y[1, 10] = 550; y[2, 6:] = 410
    np.testing.assert_array_equal(event_labels(y), [0, 1, 1])
    times = np.arange('2024-01-01', '2024-04-01', dtype='datetime64[h]')
    covered = []
    for tr, va in purged_blocks(times):
        assert np.all((times[tr] < times[va[0]] - np.timedelta64(72, 'h')) |
                      (times[tr] > times[va[-1]] + np.timedelta64(72, 'h')))
        covered.extend(va)
    np.testing.assert_array_equal(np.sort(covered), np.arange(len(times)))
