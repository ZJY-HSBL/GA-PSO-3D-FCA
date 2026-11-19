import numpy as np
import pytest
from gapso_fca.core import (Structure, Params, point_source, reference_field,
                            simulate, fca_step, ca_step, metrics)
from gapso_fca.optimize import SearchConfig, optimize

@pytest.mark.parametrize('neighborhood', [6,18,26])
@pytest.mark.parametrize('rule', [0,1,2])
def test_conservation_nonnegative(neighborhood,rule):
    state=point_source(9)
    s=Structure(neighborhood,rule,2)
    for _ in range(5):
        state=fca_step(state,s,Params(),0.02)
        assert np.all(state>=-1e-15)
        assert np.isclose(state.sum(),1.,rtol=0,atol=1e-12)


def test_ca_is_conservative_and_symmetric():
    state=point_source(11)
    for _ in range(10): state=ca_step(state,(.0016,)*3,.02)
    assert np.isclose(state.sum(),1.,atol=1e-12)
    assert np.all(state>=0)
    assert np.allclose(state,np.transpose(state,(1,0,2)))


def test_reference_normalized_and_anisotropic():
    g=reference_field(21,10,.02,(.0022,.0015,.0009))
    assert np.isclose(g.sum(),1.0,atol=1e-12)
    c=10
    assert g[c+2,c,c]>g[c,c,c+2]


def test_deterministic_optimization():
    cfg=SearchConfig(ga_population=3,ga_generations=2,pso_particles=3,pso_iterations=1)
    obj=lambda s,p:(s.neighborhood-6)**2 + s.rule + (p.dx-.0016)**2
    a=optimize('GA-PSO-FCA',obj,cfg,7)
    b=optimize('GA-PSO-FCA',obj,cfg,7)
    assert a[0] == b[0] and a[1] == b[1] and a[2] == b[2]


def test_metrics_identity():
    g=reference_field(11,3,.02,(.0016,)*3)
    assert metrics(g,g)['rmse']==0
    assert metrics(g,g)['r2']==1
