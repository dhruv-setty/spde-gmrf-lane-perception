"""SPDE precision Q(theta) = tau^2 (kappa^4 C + 2 kappa^2 G + G C^-1 G) for alpha = 2.

C is the lumped (diagonal) mass matrix and G the stiffness matrix, both assembled once.
The term G C^-1 G does not depend on theta either, so all three pieces are stored as
value arrays aligned to one fixed sparsity pattern; rebuilding Q(theta) is an axpy on the
values and never changes the pattern.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def _aligned(M: sp.spmatrix, keys: np.ndarray, n: int) -> np.ndarray:
    M = sp.coo_matrix(M)
    out = np.zeros(keys.size)
    mk = M.col.astype(np.int64) * n + M.row.astype(np.int64)
    pos = np.searchsorted(keys, mk)
    if np.any(pos >= keys.size) or np.any(keys[np.minimum(pos, keys.size - 1)] != mk):
        raise ValueError("component outside the union pattern")
    np.add.at(out, pos, M.data)
    return out


class SpdePrecision:
    def __init__(self, Cd: np.ndarray, G: sp.spmatrix):
        Cd = np.asarray(Cd, dtype=float)
        n = Cd.size
        G = sp.csc_matrix(G)
        K = (G @ sp.diags(1.0 / Cd) @ G).tocsc()
        K.eliminate_zeros()
        C = sp.diags(Cd).tocsc()
        parts = [sp.coo_matrix(m) for m in (C, G, K)]
        keys = np.unique(np.concatenate([p.col.astype(np.int64) * n + p.row.astype(np.int64) for p in parts]))
        self.n = n
        self.Cd = Cd
        self.G = G
        self.K = K
        self.indices = (keys % n).astype(np.int32)
        cols = (keys // n).astype(np.int64)
        self.indptr = np.searchsorted(cols, np.arange(n + 1)).astype(np.int32)
        self.c_vals = _aligned(C, keys, n)
        self.g_vals = _aligned(G, keys, n)
        self.k_vals = _aligned(K, keys, n)

    @property
    def nnz(self) -> int:
        return int(self.indices.size)

    def values(self, kappa: float, tau: float) -> np.ndarray:
        k2 = kappa * kappa
        return tau * tau * (k2 * k2 * self.c_vals + 2.0 * k2 * self.g_vals + self.k_vals)

    def Q(self, kappa: float, tau: float) -> sp.csc_matrix:
        return sp.csc_matrix((self.values(kappa, tau), self.indices, self.indptr), shape=(self.n, self.n))


def matern_range(kappa: float, nu: float = 1.0) -> float:
    """Practical range sqrt(8 nu) / kappa; nu = alpha - d/2 = 1 for alpha = 2 in two dimensions."""
    return float(np.sqrt(8.0 * nu) / kappa)


def marginal_variance(kappa: float, tau: float, nu: float = 1.0, d: int = 2) -> float:
    """Matern marginal variance for the continuous SPDE, Gamma(nu) / (Gamma(alpha) (4 pi)^(d/2) kappa^(2 nu) tau^2)."""
    from math import gamma, pi
    alpha = nu + d / 2.0
    return gamma(nu) / (gamma(alpha) * (4.0 * pi) ** (d / 2.0) * kappa ** (2.0 * nu) * tau * tau)
