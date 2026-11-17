import numpy as np
import pytest

from spdegmrf.cholesky import SymbolicCholesky
from spdegmrf.estimate import Estimator, dense_loglik
from spdegmrf.export import run_pipeline
from spdegmrf.fem import assemble
from spdegmrf.mesh import rect_mesh
from spdegmrf.projector import projector
from spdegmrf.spde import SpdePrecision


@pytest.fixture(scope="module")
def small():
    mesh = rect_mesh(8, 8, 1.0, 1.0)
    Cd, G = assemble(mesh)
    s = SpdePrecision(Cd, G)
    rng = np.random.default_rng(5)
    P = projector(mesh, rng.uniform(0, 1, size=(30, 2)))
    return s, P, rng


@pytest.mark.parametrize("order", ["natural", "rcm"])
def test_sparse_cholesky_matches_dense(small, order):
    s, _, rng = small
    Q = s.Q(4.0, 0.7)
    sym = SymbolicCholesky(Q, order)
    F = sym.factor(Q)
    Ad = Q.toarray()[np.ix_(sym.perm, sym.perm)]
    Ld = np.linalg.cholesky(Ad)
    assert np.allclose(F.to_sparse().toarray(), Ld, atol=1e-10)
    assert np.isclose(F.logdet(), np.linalg.slogdet(Q.toarray())[1])
    b = rng.standard_normal(Q.shape[0])
    assert np.allclose(Q @ F.solve(b), b)
    # Symbolic count equals the structural nonzeros of the dense factor.
    assert sym.nnz >= np.count_nonzero(np.abs(Ld) > 1e-14)


def test_symbolic_analysis_is_reused(small):
    s, _, _ = small
    sym = SymbolicCholesky(s.Q(1.0, 1.0))
    for kappa in (0.5, 2.0, 9.0):
        F = sym.factor(s.Q(kappa, 2.0))
        assert np.isclose(F.logdet(), np.linalg.slogdet(s.Q(kappa, 2.0).toarray())[1])
    with pytest.raises(ValueError):
        sym.factor(s.Q(1.0, 1.0)[:-1, :-1])


def test_sparse_likelihood_matches_dense(small):
    s, P, rng = small
    est = Estimator(s, P, sigma=0.3)
    y = rng.standard_normal(P.shape[0])
    for kappa, tau in ((2.0, 0.5), (6.0, 0.1)):
        ref = dense_loglik(s.Q(kappa, tau).toarray(), P.toarray(), 0.3, y)
        assert np.isclose(est.loglik(kappa, tau, y), ref, rtol=1e-9, atol=1e-8)
    assert est.analyses == 2


def test_estimation_recovers_kappa_tau():
    res = run_pipeline()
    rel_k = abs(res.kappa_hat / res.kappa_true - 1)
    rel_t = abs(res.tau_hat / res.tau_true - 1)
    print(f"\nkappa_hat={res.kappa_hat:.4f} true={res.kappa_true:.4f} rel_err={rel_k:.3f}; "
          f"tau_hat={res.tau_hat:.6f} true={res.tau_true:.6f} rel_err={rel_t:.3f}; "
          f"factorisations={res.estimator.factorisations} symbolic_analyses={res.estimator.analyses}")
    assert rel_k < 0.25 and rel_t < 0.25
    # Two symbolic analyses (Q and the posterior precision) served every factorisation.
    assert res.estimator.analyses == 2 and res.estimator.factorisations > 20
    assert np.all(np.diff(res.P_dev.indptr) == 3)
