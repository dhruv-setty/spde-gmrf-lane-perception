"""Figure 3: ordering decides the cost, on the 20 by 20 lattice."""
import numpy as np

from spdegmrf.cholesky import SymbolicCholesky
from spdegmrf.lattice import fill_in, lattice_precision


def test_lattice_precision_counts():
    assert lattice_precision(10).nnz == 1104
    Q = lattice_precision(20)
    assert Q.shape == (400, 400)
    assert Q.nnz == 4804
    assert np.diff(Q.tocsc().indptr).max() == 13


def test_fill_in_scrambled_vs_rcm():
    results = [fill_in(20, seed) for seed in range(3)]
    for r in results:
        print(f"\nseed scrambled_nnz={r.scrambled_nnz} rcm_nnz={r.rcm_nnz} reduction={r.reduction:.3f}")
        assert r.n == 400 and r.q_nnz == 4804
        # The case study reports 11,378 after RCM; this implementation reproduces it exactly.
        assert r.rcm_nnz == 11378
        # The case study reports 37,615 for its scrambled order. The count depends on the
        # random relabelling; here it is larger, so the reduction is at least the reported ~70%.
        assert r.scrambled_nnz > 30000
        assert r.reduction >= 0.70


def test_symbolic_count_matches_numeric_factor():
    Q = lattice_precision(20)
    perm = np.random.default_rng(0).permutation(400)
    sym = SymbolicCholesky(Q, perm)
    F = sym.factor(Q)
    L = np.linalg.cholesky(Q.toarray()[np.ix_(perm, perm)])
    assert np.allclose(F.to_sparse().toarray(), L, atol=1e-9)
    assert np.count_nonzero(np.abs(L) > 1e-13) <= sym.nnz
