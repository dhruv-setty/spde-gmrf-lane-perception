"""Sparse projector P from mesh nodes to sensor locations.

Row s of P holds the three barycentric weights of the triangle containing sensor s, so
the field at the sensor is w_i x_i + w_j x_j + w_k x_k with w_i + w_j + w_k = 1.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .mesh import Mesh, barycentric


def projector(mesh: Mesh, sensors: np.ndarray) -> sp.csr_matrix:
    tri, w = barycentric(mesh, sensors)
    m = w.shape[0]
    cols = mesh.tris[tri]  # (m, 3)
    rows = np.repeat(np.arange(m), 3)
    # Keep all three entries even when a weight is exactly zero (sensor on an edge or a node),
    # so every row has exactly three stored entries, as the design specifies.
    P = sp.csr_matrix((w.ravel(), (rows, cols.ravel())), shape=(m, mesh.n))
    P.sort_indices()
    return P


def posterior_precision(Q: sp.spmatrix, P: sp.spmatrix, d: np.ndarray, H: sp.spmatrix | None = None) -> sp.csc_matrix:
    """Q + P^T H^T D_S H P with D_S = diag(d)."""
    HP = sp.csr_matrix(P) if H is None else sp.csr_matrix(H) @ sp.csr_matrix(P)
    return (sp.csc_matrix(Q) + (HP.T @ sp.diags(np.asarray(d, dtype=float)) @ HP)).tocsc()
