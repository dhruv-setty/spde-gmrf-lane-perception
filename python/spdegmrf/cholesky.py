"""Sparse Cholesky factorisation with a reusable symbolic analysis.

The symbolic analysis (fill-reducing ordering, elimination tree and the column patterns
of the factor) depends only on the sparsity pattern. It is computed once and reused for
every numerical factorisation of a matrix with the same pattern, which is the situation
in the likelihood loop: Q(theta) changes its values but never its pattern.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import reverse_cuthill_mckee


def _lower_pattern(A: sp.spmatrix) -> sp.csc_matrix:
    L = sp.tril(sp.csc_matrix(A), format="csc")
    L.sort_indices()
    return L


def ordering(A: sp.spmatrix, method: str = "rcm") -> np.ndarray:
    """Return a permutation p such that A[p][:, p] is factorised."""
    n = A.shape[0]
    if method == "natural":
        return np.arange(n)
    if method == "rcm":
        return np.asarray(reverse_cuthill_mckee(sp.csr_matrix(A), symmetric_mode=True), dtype=np.int64)
    raise ValueError(f"unknown ordering {method!r}")


def symbolic_nnz(A: sp.spmatrix) -> int:
    """Number of nonzeros (diagonal included) in the Cholesky factor of A in its given order."""
    return SymbolicCholesky(A, order="natural").nnz


class SymbolicCholesky:
    """Ordering, elimination tree and column patterns of the Cholesky factor."""

    def __init__(self, A: sp.spmatrix, order: str | np.ndarray = "rcm"):
        A = sp.csc_matrix(A)
        if A.shape[0] != A.shape[1]:
            raise ValueError("matrix must be square")
        self.n = n = A.shape[0]
        self.perm = ordering(A, order) if isinstance(order, str) else np.asarray(order, dtype=np.int64)
        self.iperm = np.empty_like(self.perm)
        self.iperm[self.perm] = np.arange(n)
        Ap = _lower_pattern(A[self.perm][:, self.perm])
        self._pattern_key = self._key(A)
        # Column patterns by the elimination-tree recursion:
        # struct(L[:, j]) = struct(A[j:, j]) U (U over children c of struct(L[:, c]) minus {c}).
        parent = np.full(n, -1, dtype=np.int64)
        children: list[list[int]] = [[] for _ in range(n)]
        cols: list[np.ndarray] = []
        for j in range(n):
            s = set(Ap.indices[Ap.indptr[j]:Ap.indptr[j + 1]].tolist())
            s.add(j)
            for ch in children[j]:
                s.update(cols[ch][1:].tolist())
            idx = np.array(sorted(s), dtype=np.int64)
            cols.append(idx)
            if idx.size > 1:
                parent[j] = idx[1]
                children[idx[1]].append(j)
        self.parent = parent
        self.cols = cols
        self.nnz = int(sum(c.size for c in cols))
        # Row lists: for row i, the columns k < i holding L[i, k] and the position of i in col k.
        rows: list[list[tuple[int, int]]] = [[] for _ in range(n)]
        for k, idx in enumerate(cols):
            for pos in range(1, idx.size):
                rows[int(idx[pos])].append((k, pos))
        self.rows = rows

    @staticmethod
    def _key(A: sp.csc_matrix):
        L = _lower_pattern(A)
        return (A.shape, L.indptr.tobytes(), L.indices.tobytes())

    def same_pattern(self, A: sp.spmatrix) -> bool:
        return self._key(sp.csc_matrix(A)) == self._pattern_key

    def factor(self, A: sp.spmatrix) -> "CholeskyFactor":
        return CholeskyFactor(self, A)


class CholeskyFactor:
    """Numerical factor L with P A P^T = L L^T, computed on a SymbolicCholesky."""

    def __init__(self, sym: SymbolicCholesky, A: sp.spmatrix):
        A = sp.csc_matrix(A)
        if not sym.same_pattern(A):
            raise ValueError("matrix pattern differs from the analysed pattern")
        self.sym = sym
        n = sym.n
        Ap = _lower_pattern(A[sym.perm][:, sym.perm])
        w = np.zeros(n)
        vals: list[np.ndarray] = []
        for j in range(n):
            idx = sym.cols[j]
            w[idx] = 0.0
            lo, hi = Ap.indptr[j], Ap.indptr[j + 1]
            w[Ap.indices[lo:hi]] = Ap.data[lo:hi]
            for k, pos in sym.rows[j]:
                vk = vals[k]
                w[sym.cols[k][pos:]] -= vk[pos:] * vk[pos]
            d = w[j]
            if not d > 0.0:
                raise np.linalg.LinAlgError(f"matrix is not positive definite (pivot {j})")
            ljj = np.sqrt(d)
            v = w[idx] / ljj
            v[0] = ljj
            vals.append(v)
        self.vals = vals

    def diag(self) -> np.ndarray:
        return np.array([v[0] for v in self.vals])

    def logdet(self) -> float:
        """log|A| = 2 sum log L_ii."""
        return float(2.0 * np.sum(np.log(self.diag())))

    def to_sparse(self) -> sp.csc_matrix:
        """L in the permuted order."""
        n = self.sym.n
        indptr = np.cumsum([0] + [c.size for c in self.sym.cols])
        return sp.csc_matrix((np.concatenate(self.vals), np.concatenate(self.sym.cols), indptr), shape=(n, n))

    def _lsolve(self, b: np.ndarray) -> np.ndarray:
        x = b.astype(float).copy()
        for j in range(self.sym.n):
            v, idx = self.vals[j], self.sym.cols[j]
            x[j] /= v[0]
            x[idx[1:]] -= np.multiply.outer(v[1:], x[j]) if x.ndim > 1 else v[1:] * x[j]
        return x

    def _ltsolve(self, b: np.ndarray) -> np.ndarray:
        x = b.astype(float).copy()
        for j in range(self.sym.n - 1, -1, -1):
            v, idx = self.vals[j], self.sym.cols[j]
            x[j] -= v[1:] @ x[idx[1:]]
            x[j] /= v[0]
        return x

    def solve(self, b: np.ndarray) -> np.ndarray:
        """Solve A x = b."""
        b = np.asarray(b, dtype=float)
        y = self._lsolve(b[self.sym.perm])
        z = self._ltsolve(y)
        return z[self.sym.iperm]

    def solve_lt(self, z: np.ndarray) -> np.ndarray:
        """Return x with L^T (P x) = z, so x ~ N(0, A^-1) when z ~ N(0, I)."""
        return self._ltsolve(np.asarray(z, dtype=float))[self.sym.iperm]
