import numpy as np
import scipy.sparse as sp

from spdegmrf.fem import assemble
from spdegmrf.mesh import rect_mesh, scrambled
from spdegmrf.projector import posterior_precision, projector
from spdegmrf.spde import SpdePrecision


def test_fem_unit_lattice_is_five_point_laplacian():
    Cd, G = assemble(rect_mesh(5, 5))
    # Interior node: lumped mass 1 (six triangles of area 1/2, a third each), stiffness 4.
    c = 2 * 5 + 2
    assert np.isclose(Cd[c], 1.0)
    assert np.isclose(G[c, c], 4.0)
    assert sorted(np.round(G[:, c].toarray().ravel()[G[:, c].toarray().ravel() != 0], 12)) == [-1, -1, -1, -1, 4]
    # G annihilates constants; total mass equals the area.
    assert np.allclose(G @ np.ones(25), 0.0)
    assert np.isclose(Cd.sum(), 16.0)


def test_spde_precision_matches_formula_and_is_spd():
    mesh = rect_mesh(6, 7, 1.0, 1.2)
    Cd, G = assemble(mesh)
    s = SpdePrecision(Cd, G)
    kappa, tau = 3.0, 0.5
    Cinv = sp.diags(1.0 / Cd)
    ref = tau ** 2 * (kappa ** 4 * sp.diags(Cd) + 2 * kappa ** 2 * G + G @ Cinv @ G)
    Q = s.Q(kappa, tau)
    assert np.allclose(Q.toarray(), ref.toarray())
    assert np.allclose(Q.toarray(), Q.toarray().T)
    assert np.linalg.eigvalsh(Q.toarray()).min() > 0
    # The pattern is fixed: same indices for any theta.
    Q2 = s.Q(0.3, 7.0)
    assert np.array_equal(Q.indices, Q2.indices) and np.array_equal(Q.indptr, Q2.indptr)


def test_projector_three_weights_per_row_summing_to_one():
    mesh = scrambled(rect_mesh(9, 9, 2.0, 2.0), seed=3)
    rng = np.random.default_rng(0)
    pts = np.vstack([rng.uniform(0, 2, size=(50, 2)), [[0.0, 0.0], [2.0, 2.0], [1.0, 1.0], [0.25, 0.0]]])
    P = projector(mesh, pts)
    nnz_per_row = np.diff(P.indptr)
    assert np.all(nnz_per_row == 3)
    assert np.allclose(np.asarray(P.sum(axis=1)).ravel(), 1.0)
    assert P.data.min() >= 0.0
    # Linear functions are reproduced exactly: P applied to node coordinates gives the sensors.
    assert np.allclose(P @ mesh.nodes, pts)
    # The three columns of each row are the corners of one triangle.
    tri_sets = {frozenset(t) for t in mesh.tris.tolist()}
    for s in range(P.shape[0]):
        assert frozenset(P.indices[P.indptr[s]:P.indptr[s + 1]].tolist()) in tri_sets


def test_posterior_precision_stays_sparse_and_local():
    mesh = rect_mesh(12, 12, 1.0, 1.0)
    Cd, G = assemble(mesh)
    Q = SpdePrecision(Cd, G).Q(5.0, 1.0)
    rng = np.random.default_rng(1)
    P = projector(mesh, rng.uniform(0, 1, size=(40, 2)))
    Qp = posterior_precision(Q, P, np.full(40, 4.0))
    assert np.allclose(Qp.toarray(), Qp.toarray().T)
    added = (sp.csc_matrix(abs(Qp)) > 0).astype(int) - (sp.csc_matrix(abs(Q)) > 0).astype(int)
    added.eliminate_zeros()
    # Lifting only couples nodes of a shared triangle, which are already neighbours in Q.
    assert added.nnz == 0
    assert Qp.nnz == Q.nnz
    assert Qp.nnz < 0.1 * Qp.shape[0] ** 2
