"""Figure 4: int8 forward filter checked against the float reference."""
import ctypes
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from spdegmrf import hmm, quant

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def run():
    rng = np.random.default_rng(7)
    p = hmm.HmmParams.default(rng.normal(0.0, 40.0, 64))
    truth, sites, Y = hmm.simulate(p, 5000, rng)
    fb = hmm.forward_filter(p, Y, sites)
    m = quant.quantise(p)
    f = quant.Int8Filter(m)
    qb, modes = f.run(quant.quantise_obs(Y), sites)
    return p, m, truth, sites, Y, fb, qb, modes, f


def test_transition_matrix_keeps_direct_calm_to_panic(run):
    p, m, *_ = run
    assert p.A[0, 3] > 0
    assert m.W[3, 0] > 0  # weight into panic from calm survives int8 quantisation


def test_exp_table_matches_committed_c_source():
    src = (ROOT / "firmware/src/hmm_exp_lut.c").read_text()
    body = src[src.index("{") + 1:src.index("}")]
    assert [int(v) for v in body.replace("\n", " ").split(",") if v.strip()] == quant.EXP2_LUT


def test_int8_agrees_with_float(run):
    *_, truth, sites, Y, fb, qb, modes, f = run
    ag = quant.agreement(fb, qb, truth)
    print(f"\nint8 vs float over T={ag.T}: argmax agreement={ag.argmax_agreement:.4f}, "
          f"mean |belief error|={ag.mean_abs_belief_error:.4f}, max={ag.max_abs_belief_error:.4f}; "
          f"accuracy vs simulated truth float={ag.float_accuracy_vs_truth:.4f} "
          f"int8={ag.int8_accuracy_vs_truth:.4f}; floor hits={f.floor_hits}")
    assert ag.argmax_agreement >= 0.98
    assert ag.mean_abs_belief_error < 0.01
    assert np.array_equal(modes, np.asarray(qb).argmax(1))


def test_belief_never_underflows():
    p = hmm.HmmParams.default(np.zeros(1))
    f = quant.Int8Filter(quant.quantise(p))
    for t in range(300):
        y = (32767, 32767) if t % 2 else (-32768, 0)
        f.step(y, 0)
        s = sum(v + 128 for v in f.belief)
        assert 253 <= s <= 257


def _build_lib(tmp_path):
    cc = os.environ.get("CC") or shutil.which("cc") or shutil.which("gcc")
    if not cc:
        pytest.skip("no C compiler")
    fw = ROOT / "firmware"
    lib = tmp_path / "libhmmfilter.so"
    srcs = [str(fw / "src" / s) for s in ("hmm_filter.c", "hmm_fc_s8_ref.c", "hmm_exp_lut.c")]
    subprocess.run([cc, "-std=c99", "-O2", "-fPIC", "-shared", "-I", str(fw / "include"), "-o", str(lib), *srcs],
                   check=True)
    return ctypes.CDLL(str(lib))


class _Model(ctypes.Structure):
    _fields_ = [("w", ctypes.c_int8 * 16), ("bias", ctypes.c_int32 * 4), ("fc_multiplier", ctypes.c_int32),
                ("fc_shift", ctypes.c_int32), ("input_offset", ctypes.c_int32), ("output_offset", ctypes.c_int32),
                ("mu", ctypes.c_int32 * 8), ("inv_sigma_q16", ctypes.c_int32 * 8), ("lognorm_q16", ctypes.c_int32 * 4),
                ("offset_channel_mask", ctypes.c_uint8), ("site_offset_mm", ctypes.POINTER(ctypes.c_int16)),
                ("n_sites", ctypes.c_uint16), ("init_belief", ctypes.c_int8 * 4)]


class _State(ctypes.Structure):
    _fields_ = [("model", ctypes.POINTER(_Model)), ("belief", ctypes.c_int8 * 4),
                ("steps", ctypes.c_uint32), ("floor_hits", ctypes.c_uint32)]


def test_c_kernel_is_bit_exact_with_python_mirror(run, tmp_path):
    p, m, truth, sites, Y, fb, qb, modes, _ = run
    lib = _build_lib(tmp_path)
    offs = (ctypes.c_int16 * len(m.site_offset_mm))(*m.site_offset_mm.tolist())
    cm = _Model((ctypes.c_int8 * 16)(*m.W.ravel().tolist()), (ctypes.c_int32 * 4)(*m.bias.tolist()),
                m.mult, m.shift, m.input_offset, m.output_offset,
                (ctypes.c_int32 * 8)(*m.mu.ravel().tolist()), (ctypes.c_int32 * 8)(*m.inv_sigma_q16.ravel().tolist()),
                (ctypes.c_int32 * 4)(*m.lognorm_q16.tolist()), m.offset_mask, offs, len(m.site_offset_mm),
                (ctypes.c_int8 * 4)(*m.init_belief.tolist()))
    st = _State()
    assert lib.hmm_init(ctypes.byref(st), ctypes.byref(cm)) == 0
    Yq = quant.quantise_obs(Y)
    for t in range(len(Yq)):
        y = (ctypes.c_int16 * 2)(*Yq[t].tolist())
        mode = lib.hmm_step(ctypes.byref(st), y, ctypes.c_uint16(int(sites[t])))
        assert mode == modes[t]
        assert list(st.belief) == qb[t].tolist()
