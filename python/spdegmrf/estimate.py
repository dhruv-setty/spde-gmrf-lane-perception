"""Gaussian marginal likelihood for y = P x + eps and estimation of theta = (kappa, tau).

x ~ N(0, Q(theta)^-1), eps ~ N(0, sigma^2 I), with R independent replicates as the columns
of Y. With Q_p = Q + P^T P / sigma^2 and mu = Q_p^-1 P^T y / sigma^2,

  log p(y) = 1/2 log|Q| - 1/2 log|Q_p| - m/2 log(2 pi sigma^2)
             - 1/2 (y^T y / sigma^2 - mu^T Q_p mu).

Both log-determinants come from 2 sum log L_ii of the respective Cholesky factors, and
both symbolic analyses are made once and reused for every theta.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize

from .cholesky import SymbolicCholesky
from .spde import SpdePrecision


@dataclass
class Estimator:
    spde: SpdePrecision
    P: sp.csr_matrix
    sigma: float
    order: str = "rcm"
    analyses: int = field(default=0, init=False)
    factorisations: int = field(default=0, init=False)

    def __post_init__(self):
        self.P = sp.csr_matrix(self.P)
        self.PtP = (self.P.T @ self.P).tocsc()
        Q0 = self.spde.Q(1.0, 1.0)
        self._symQ = SymbolicCholesky(Q0, self.order)
        self._symQp = SymbolicCholesky(self._qp(Q0), self.order)
        self.analyses = 2

    def _qp(self, Q: sp.csc_matrix) -> sp.csc_matrix:
        Qp = (Q + self.PtP / self.sigma ** 2).tocsc()
        Qp.sort_indices()
        return Qp

    def loglik(self, kappa: float, tau: float, Y: np.ndarray) -> float:
        Y = np.asarray(Y, dtype=float)
        if Y.ndim == 1:
            Y = Y[:, None]
        m, R = Y.shape
        Q = self.spde.Q(kappa, tau)
        Qp = self._qp(Q)
        FQ = self._symQ.factor(Q)
        FQp = self._symQp.factor(Qp)
        self.factorisations += 2
        b = (self.P.T @ Y) / self.sigma ** 2
        mu = FQp.solve(b)
        quad = np.sum(Y * Y) / self.sigma ** 2 - np.sum(b * mu)
        return float(0.5 * R * (FQ.logdet() - FQp.logdet())
                     - 0.5 * m * R * np.log(2.0 * np.pi * self.sigma ** 2) - 0.5 * quad)

    def posterior_mean(self, kappa: float, tau: float, Y: np.ndarray) -> np.ndarray:
        Qp = self._qp(self.spde.Q(kappa, tau))
        F = self._symQp.factor(Qp)
        self.factorisations += 1
        return F.solve((self.P.T @ np.asarray(Y, dtype=float)) / self.sigma ** 2)

    def fit(self, Y: np.ndarray, kappa0: float = 0.5, tau0: float = 1.0):
        """Maximise the marginal likelihood over (log kappa, log tau). Returns (kappa, tau, loglik)."""
        f = lambda z: -self.loglik(np.exp(z[0]), np.exp(z[1]), Y)
        res = minimize(f, x0=np.log([kappa0, tau0]), method="Nelder-Mead",
                       options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 400})
        return float(np.exp(res.x[0])), float(np.exp(res.x[1])), float(-res.fun)


def sample_field(spde: SpdePrecision, kappa: float, tau: float, rng: np.random.Generator, size: int = 1) -> np.ndarray:
    """Draw x ~ N(0, Q^-1) via L^T x = z."""
    F = SymbolicCholesky(spde.Q(kappa, tau), "rcm").factor(spde.Q(kappa, tau))
    z = rng.standard_normal((spde.n, size))
    return F.solve_lt(z)


def dense_loglik(Q: np.ndarray, P: np.ndarray, sigma: float, y: np.ndarray) -> float:
    """Reference: y ~ N(0, P Q^-1 P^T + sigma^2 I), evaluated densely."""
    S = P @ np.linalg.solve(Q, P.T) + sigma ** 2 * np.eye(P.shape[0])
    sign, ld = np.linalg.slogdet(S)
    return float(-0.5 * (ld + y @ np.linalg.solve(S, y) + y.size * np.log(2 * np.pi)))
