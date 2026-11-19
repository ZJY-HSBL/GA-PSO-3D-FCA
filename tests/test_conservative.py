"""Regression tests for the proposed conservative-flux research extension."""
import numpy as np
import pytest
from gapso_fca.core import Params, Structure, ca_step, fca_step, point_source
from gapso_fca.conservative import conservative_fca_step, conservative_simulate


@pytest.mark.parametrize('rule', [0, 1, 2])
def test_pairwise_conservation_and_maximum_principle(rule):
    rng = np.random.default_rng(2026)
    state = rng.random((8, 8, 8))
    initial_min, initial_max = state.min(), state.max()
    mass = state.sum()
    for _ in range(12):
        state, info = conservative_fca_step(state, (.0022, .0015, .0009), .04,
                                            rule=rule, strength=1, return_diagnostics=True)
        assert info.maximum_courant <= .95 + 1e-14
        assert state.min() >= initial_min - 1e-12
        assert state.max() <= initial_max + 1e-12
        assert abs(state.sum() - mass) < 1e-10


def test_exact_ordinary_ca_recovery():
    state = point_source(15)
    expected = ca_step(state, (.0016,)*3, .02)
    observed = conservative_fca_step(state, (.0016,)*3, .02, strength=0)
    assert np.allclose(observed, expected, atol=1e-16, rtol=0)


def test_zero_gradient_uniform_state_fixed():
    u = np.full((9, 9, 9), .4)
    result = conservative_fca_step(u, (.0028, .0021, .0012), .2, rule=2)
    assert np.array_equal(result, u)


def test_tiny_gradient_flux_vanishes_old_rule_does_not():
    old_change, new_change = [], []
    for eps in (1e-4, 1e-7, 1e-10):
        u = np.ones((9,)*3)
        u[4, 4, 4] -= eps
        previous = fca_step(u, Structure(), Params(), .02)
        proposed = conservative_fca_step(u, (.0016,)*3, .02)
        old_change.append(np.max(np.abs(previous-u)))
        new_change.append(np.max(np.abs(proposed-u)))
    assert old_change[-1] > 1e-4  # discontinuous direction normalization
    assert new_change[-1] < 1e-9  # consistent with vanishing gradient
    assert new_change[-1] < new_change[0] * 1e-5


def test_mask_and_material_interface():
    n = 12
    state = np.zeros((n, n, n))
    state[2, 6, 6] = 1.0
    active = np.ones_like(state, dtype=bool)
    active[6, :, :] = False
    d = np.full((3, n, n, n), .001)
    d[:, :6] = .002
    d[:, 7:] = .0005
    saved = conservative_simulate(state, d, .02, 30, mask=active, snapshots=(30,))
    end = saved[30]
    assert np.isclose(end.sum(), 1., atol=1e-13)
    assert np.all(end[6:] == 0)  # impermeable barrier


def test_adaptive_substeps_and_nonnegative():
    u = point_source(11)
    out, info = conservative_fca_step(u, (.0028,)*3, 1.2, strength=1,
                                      return_diagnostics=True)
    assert info.substeps > 1
    assert out.min() >= -1e-14
    assert np.isclose(out.sum(), 1., atol=1e-13)


def test_reject_invalid_materials_and_mask():
    u = point_source(5)
    with pytest.raises(ValueError):
        conservative_fca_step(u, (-.001, .002, .003), .02)
    with pytest.raises(ValueError):
        conservative_fca_step(u, (.002,)*3, .02, mask=np.zeros_like(u, bool))
