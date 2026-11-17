"""int8 quantisation of the forward filter and a bit-exact integer mirror of the C kernel.

Belief representation
    int8 q in [-128, 127] stands for (q + 128) / 255 (scale 1/255, zero point -128).

Prediction  A^T alpha  as a fully connected layer with arm_fully_connected_s8 semantics
    weights   W[k][j] = round(A[j][k] / s_w), s_w = max(A) / 127, symmetric (filter offset 0)
    input     offset +128, output offset -128, bias 0, activation range [-128, 127]
    requant   multiplier/shift for s_in * s_w / s_out = s_w, CMSIS-NN default rounding
              (doubling high multiply then rounding divide by a power of two).

Emission    log b_k(y) in Q16 from per-regime integer means (mm), inverse standard deviations
            in Q16 and a Q16 log-normaliser; exp(-(max - ll_k)) through a 256-entry Q15 table
            of 2^(-i/256) and a right shift, floored at 1 so it never reaches zero.

Update      u_k = max(pred_k + 128, 1) * e_k; q_k = round(255 u_k / sum u) - 128.
            The two floors keep the belief from underflowing; argmax of q is the mode.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .hmm import D, K, OFFSET_CHANNEL, HmmParams

Q16 = 1 << 16
LOG2E_Q16 = 94548  # round(log2(e) * 2^16)
Z_CLAMP_Q16 = 1 << 24  # |z| <= 256 standard deviations before squaring
EXP2_LUT = [int(round(32768.0 * 2.0 ** (-i / 256.0))) for i in range(256)]
INT32_MAX = 2 ** 31 - 1


def quantize_multiplier(real: float):
    """TFLite / CMSIS-NN style (multiplier Q31, shift) with real = multiplier * 2^(shift - 31)."""
    if real <= 0.0:
        return 0, 0
    q, shift = np.frexp(real)
    qf = int(round(q * (1 << 31)))
    if qf == (1 << 31):
        qf //= 2
        shift += 1
    return int(qf), int(shift)


def _wrap32(v: int) -> int:
    return ((v + 2 ** 31) % 2 ** 32) - 2 ** 31


def doubling_high_mult_no_sat(m1: int, m2: int) -> int:
    return _wrap32((m1 * m2 + (1 << 30)) >> 31)


def divide_by_power_of_two(dividend: int, exponent: int) -> int:
    mask = (1 << exponent) - 1
    remainder = dividend & mask
    result = dividend >> exponent
    threshold = mask >> 1
    if result < 0:
        threshold += 1
    if remainder > threshold:
        result += 1
    return result


def requantize(val: int, multiplier: int, shift: int) -> int:
    left = shift if shift > 0 else 0
    right = -shift if shift < 0 else 0
    return divide_by_power_of_two(doubling_high_mult_no_sat(_wrap32(val * (1 << left)), multiplier), right)


def fc_s8(inp, W, bias, in_off, out_off, mult, shift, act_min=-128, act_max=127):
    """Portable reference of arm_fully_connected_s8 for one batch."""
    out = []
    for k in range(len(W)):
        acc = int(bias[k])
        for j in range(len(inp)):
            acc += (int(inp[j]) + in_off) * int(W[k][j])
        r = requantize(acc, mult, shift) + out_off
        out.append(max(act_min, min(act_max, r)))
    return out


@dataclass
class Int8Model:
    W: np.ndarray            # (K, K) int8, row k = weights into regime k (A^T)
    bias: np.ndarray         # (K,) int32
    mult: int
    shift: int
    input_offset: int
    output_offset: int
    mu: np.ndarray           # (K, D) int32 mm
    inv_sigma_q16: np.ndarray  # (K, D) int32
    lognorm_q16: np.ndarray  # (K,) int32
    offset_mask: int         # bit d set: channel d receives the lifted offset
    site_offset_mm: np.ndarray  # (n_sites,) int16
    init_belief: np.ndarray  # (K,) int8
    s_w: float


def quantise(p: HmmParams) -> Int8Model:
    p.validate()
    s_w = float(p.A.max()) / 127.0
    W = np.clip(np.round(p.A.T / s_w), -127, 127).astype(np.int8)
    mult, shift = quantize_multiplier(s_w)
    init = np.clip(np.round(255.0 * p.pi0) - 128, -128, 127).astype(np.int8)
    return Int8Model(
        W=W, bias=np.zeros(K, np.int32), mult=mult, shift=shift,
        input_offset=128, output_offset=-128,
        mu=np.round(p.mu).astype(np.int32),
        inv_sigma_q16=np.round(Q16 / p.sigma).astype(np.int32),
        lognorm_q16=np.round(-np.sum(np.log(p.sigma), axis=1) * Q16).astype(np.int32),
        offset_mask=1 << OFFSET_CHANNEL,
        site_offset_mm=np.clip(np.round(p.site_offset), -32768, 32767).astype(np.int16),
        init_belief=init, s_w=s_w,
    )


def quantise_obs(y: np.ndarray) -> np.ndarray:
    return np.clip(np.round(np.asarray(y, float)), -32768, 32767).astype(np.int16)


def exp_neg_q15(delta_q16: int) -> int:
    """exp(-delta) in Q15 for delta >= 0 in Q16, floored at 1."""
    t = (delta_q16 * LOG2E_Q16) >> 16
    ip = t >> 16
    if ip >= 15:
        return 1
    frac = (t >> 8) & 0xFF
    return max(EXP2_LUT[frac] >> ip, 1)


class Int8Filter:
    """Integer mirror of firmware/src/hmm_filter.c; results must match bit for bit."""

    def __init__(self, m: Int8Model):
        self.m = m
        self.belief = [int(v) for v in m.init_belief]
        self.floor_hits = 0

    def step(self, y_mm, site: int) -> int:
        m = self.m
        pred = fc_s8(self.belief, m.W.tolist(), m.bias.tolist(), m.input_offset, m.output_offset, m.mult, m.shift)
        ll = []
        for k in range(K):
            acc = int(m.lognorm_q16[k])
            sq = 0
            for d in range(D):
                mean = int(m.mu[k][d])
                if (m.offset_mask >> d) & 1:
                    mean += int(m.site_offset_mm[site])
                z = (int(y_mm[d]) - mean) * int(m.inv_sigma_q16[k][d])
                z = max(-Z_CLAMP_Q16, min(Z_CLAMP_Q16, z))
                sq += (z * z) >> 16
            ll.append(acc - (sq >> 1))
        mx = max(ll)
        u = []
        for k in range(K):
            pk = pred[k] + 128
            if pk < 1:
                pk = 1
                self.floor_hits += 1
            e = exp_neg_q15(mx - ll[k])
            u.append(pk * e)
        s = sum(u)
        self.belief = [max(-128, min(127, (255 * uk + s // 2) // s - 128)) for uk in u]
        best = 0
        for k in range(1, K):
            if self.belief[k] > self.belief[best]:
                best = k
        return best

    def run(self, Yq: np.ndarray, sites: np.ndarray):
        beliefs = np.empty((len(Yq), K), dtype=np.int8)
        modes = np.empty(len(Yq), dtype=np.int64)
        for t in range(len(Yq)):
            modes[t] = self.step(Yq[t], int(sites[t]))
            beliefs[t] = self.belief
        return beliefs, modes


def dequant_belief(q: np.ndarray) -> np.ndarray:
    return (np.asarray(q, float) + 128.0) / 255.0


@dataclass
class Agreement:
    T: int
    argmax_agreement: float
    mean_abs_belief_error: float
    max_abs_belief_error: float
    float_accuracy_vs_truth: float
    int8_accuracy_vs_truth: float


def agreement(float_beliefs, int8_beliefs, truth=None) -> Agreement:
    fb = np.asarray(float_beliefs)
    qb = dequant_belief(int8_beliefs)
    qn = qb / qb.sum(1, keepdims=True)
    fa = fb.argmax(1)
    # Tie-break as the C kernel does: first maximum of the int8 belief.
    qa = np.asarray(int8_beliefs).argmax(1)
    err = np.abs(qn - fb)
    acc_f = float(np.mean(fa == truth)) if truth is not None else float("nan")
    acc_q = float(np.mean(qa == truth)) if truth is not None else float("nan")
    return Agreement(T=len(fb), argmax_agreement=float(np.mean(fa == qa)),
                     mean_abs_belief_error=float(err.mean()), max_abs_belief_error=float(err.max()),
                     float_accuracy_vs_truth=acc_f, int8_accuracy_vs_truth=acc_q)
