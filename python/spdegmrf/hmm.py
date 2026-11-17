"""Four-regime hidden Markov model and its floating-point forward filter.

Regimes: 0 calm, 1 alert, 2 high-stress, 3 panic. Observations are two channels per
sample: lateral lane offset (mm) and the SDLP over a short window (mm). The lifted field
offset (P x_hat)_s for the current sensor site is added to the mean of the lateral offset
channel (H selects channel 0).

All parameter values here are illustrative. They have not been fitted to recorded data.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

REGIMES = ("calm", "alert", "high-stress", "panic")
K = 4
D = 2

#: Transition matrix A, rows sum to one; A[0, 3] is the direct calm -> panic path.
A_DEFAULT = np.array([
    [0.940, 0.040, 0.015, 0.005],
    [0.060, 0.880, 0.050, 0.010],
    [0.010, 0.080, 0.860, 0.050],
    [0.010, 0.020, 0.120, 0.850],
])

#: Emission means (mm): lateral offset, SDLP window.
MU_DEFAULT = np.array([
    [0.0, 120.0],
    [0.0, 180.0],
    [30.0, 280.0],
    [80.0, 420.0],
])

#: Emission standard deviations (mm).
SIGMA_DEFAULT = np.array([
    [60.0, 25.0],
    [80.0, 30.0],
    [110.0, 45.0],
    [160.0, 70.0],
])

#: Channel that receives the lifted field offset (the selector H).
OFFSET_CHANNEL = 0


@dataclass
class HmmParams:
    A: np.ndarray
    mu: np.ndarray
    sigma: np.ndarray
    pi0: np.ndarray
    site_offset: np.ndarray  # (n_sites,) lifted offsets P x_hat in mm

    @staticmethod
    def default(site_offset: np.ndarray | None = None) -> "HmmParams":
        A = A_DEFAULT.copy()
        return HmmParams(A=A, mu=MU_DEFAULT.copy(), sigma=SIGMA_DEFAULT.copy(),
                         pi0=stationary(A),
                         site_offset=np.zeros(1) if site_offset is None else np.asarray(site_offset, float))

    def validate(self) -> None:
        assert self.A.shape == (K, K) and np.allclose(self.A.sum(1), 1.0) and np.all(self.A > 0)
        assert self.mu.shape == (K, D) and self.sigma.shape == (K, D) and np.all(self.sigma > 0)


def stationary(A: np.ndarray) -> np.ndarray:
    w, v = np.linalg.eig(A.T)
    p = np.real(v[:, np.argmin(np.abs(w - 1.0))])
    return p / p.sum()


def emission_loglik(p: HmmParams, y: np.ndarray, site: int) -> np.ndarray:
    mean = p.mu.copy()
    mean[:, OFFSET_CHANNEL] += p.site_offset[site]
    z = (np.asarray(y, float)[None, :] - mean) / p.sigma
    return -0.5 * np.sum(z * z, axis=1) - np.sum(np.log(p.sigma), axis=1) - 0.5 * D * np.log(2 * np.pi)


def forward_filter(p: HmmParams, Y: np.ndarray, sites: np.ndarray) -> np.ndarray:
    """alpha_t(k) proportional to b_k(y_t) sum_j a_jk alpha_{t-1}(j); returns (T, K) beliefs."""
    alpha = p.pi0.copy()
    out = np.empty((len(Y), K))
    for t, (y, s) in enumerate(zip(Y, sites)):
        pred = p.A.T @ alpha
        ll = emission_loglik(p, y, int(s))
        a = pred * np.exp(ll - ll.max())
        alpha = a / a.sum()
        out[t] = alpha
    return out


def simulate(p: HmmParams, T: int, rng: np.random.Generator, n_sites: int | None = None):
    """Simulate regimes, sites and observations. Sites walk forward along a route."""
    n_sites = len(p.site_offset) if n_sites is None else n_sites
    s = rng.choice(K, p=p.pi0)
    states = np.empty(T, dtype=np.int64)
    sites = (np.arange(T) * n_sites // T).astype(np.int64)
    Y = np.empty((T, D))
    for t in range(T):
        if t > 0:
            s = rng.choice(K, p=p.A[s])
        states[t] = s
        mean = p.mu[s].copy()
        mean[OFFSET_CHANNEL] += p.site_offset[sites[t]]
        Y[t] = mean + p.sigma[s] * rng.standard_normal(D)
    return states, sites, Y
