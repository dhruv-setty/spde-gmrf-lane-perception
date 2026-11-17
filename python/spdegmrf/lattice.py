"""Figure 3 experiment: Cholesky fill-in on a 20 by 20 lattice under two node orders."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cholesky import SymbolicCholesky
from .fem import assemble
from .mesh import rect_mesh
from .spde import SpdePrecision


@dataclass
class FillIn:
    n: int
    q_nnz: int
    scrambled_nnz: int
    rcm_nnz: int

    @property
    def reduction(self) -> float:
        return 1.0 - self.rcm_nnz / self.scrambled_nnz


def lattice_precision(side: int = 20, kappa: float = 1.0, tau: float = 1.0):
    Cd, G = assemble(rect_mesh(side, side))
    return SpdePrecision(Cd, G).Q(kappa, tau)


def fill_in(side: int = 20, seed: int = 0) -> FillIn:
    """Factor nonzeros (diagonal included) for a scrambled node order and for RCM.

    The scrambled order is a uniformly random relabelling with the given seed.
    """
    Q = lattice_precision(side)
    n = Q.shape[0]
    perm = np.random.default_rng(seed).permutation(n)
    scr = SymbolicCholesky(Q, perm).nnz
    # RCM is applied to the scrambled matrix, as it would be to a mesh generator's output.
    Qs = Q[perm][:, perm]
    rcm = SymbolicCholesky(Qs, "rcm").nnz
    return FillIn(n=n, q_nnz=int(Q.nnz), scrambled_nnz=int(scr), rcm_nnz=int(rcm))
