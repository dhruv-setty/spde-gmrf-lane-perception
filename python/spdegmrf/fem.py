"""P1 finite-element assembly: lumped mass matrix C and stiffness matrix G."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .mesh import Mesh

DROP_TOL = 1e-12


def assemble(mesh: Mesh):
    """Return (C_lumped_diagonal, G) for piecewise-linear elements.

    C is returned as a 1-D array (the diagonal of the lumped mass matrix), G as CSC.
    Entries of G whose magnitude falls below DROP_TOL relative to its largest entry are
    removed, so cancelling couplings do not count as structural nonzeros.
    """
    n = mesh.n
    p = mesh.nodes[mesh.tris]  # (m, 3, 2)
    x, y = p[..., 0], p[..., 1]
    b = np.stack([y[:, 1] - y[:, 2], y[:, 2] - y[:, 0], y[:, 0] - y[:, 1]], axis=1)
    c = np.stack([x[:, 2] - x[:, 1], x[:, 0] - x[:, 2], x[:, 1] - x[:, 0]], axis=1)
    area = 0.5 * np.abs(b[:, 0] * c[:, 1] - b[:, 1] * c[:, 0])
    if np.any(area <= 0):
        raise ValueError("degenerate triangle")
    ke = (b[:, :, None] * b[:, None, :] + c[:, :, None] * c[:, None, :]) / (4.0 * area[:, None, None])
    rows = np.repeat(mesh.tris, 3, axis=1).ravel()
    cols = np.tile(mesh.tris, (1, 3)).ravel()
    G = sp.coo_matrix((ke.ravel(), (rows, cols)), shape=(n, n)).tocsc()
    G.sum_duplicates()
    tol = DROP_TOL * np.abs(G.data).max()
    G.data[np.abs(G.data) < tol] = 0.0
    G.eliminate_zeros()
    Cd = np.zeros(n)
    np.add.at(Cd, mesh.tris.ravel(), np.repeat(area / 3.0, 3))
    return Cd, G
