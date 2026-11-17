"""The GPU input file is consistent with the CPU reference it is compared against."""
import struct

import numpy as np
import scipy.sparse as sp

from spdegmrf.export import run_pipeline, write_gpu_case


def test_gpu_case_round_trip(tmp_path):
    res = run_pipeline()
    path = tmp_path / "case.bin"
    write_gpu_case(path, res)
    raw = path.read_bytes()
    assert raw[:8] == b"SPDECASE"
    n, nnz, m, pnnz, R, sigma = struct.unpack_from("<iiiiid", raw, 8)
    off = 8 + 28
    def take(dt, count):
        nonlocal off
        a = np.frombuffer(raw, dtype=dt, count=count, offset=off)
        off += a.nbytes
        return a
    qptr, qidx = take("<i4", n + 1), take("<i4", nnz)
    c, g, k = take("<f8", nnz), take("<f8", nnz), take("<f8", nnz)
    pptr, pidx, pval = take("<i4", m + 1), take("<i4", pnnz), take("<f8", pnnz)
    y = take("<f8", m * R).reshape((m, R), order="F")
    assert off == len(raw)
    kappa, tau = 4.7, 0.0015
    Q = sp.csr_matrix((tau ** 2 * (kappa ** 4 * c + 2 * kappa ** 2 * g + k), qidx, qptr), shape=(n, n))
    assert np.allclose(Q.toarray(), res.estimator.spde.Q(kappa, tau).toarray())
    P = sp.csr_matrix((pval, pidx, pptr), shape=(m, n))
    # The GPU path assumes P^T P lies inside the pattern of Q, so Q_p reuses Q's analysis.
    ptp = (P.T @ P).tocsr()
    assert ((abs(ptp) > 0).astype(int) - (abs(ptp) > 0).multiply(abs(Q) > 0).astype(int)).nnz == 0
    assert np.allclose(y, res.Y_train)
    assert sigma == res.estimator.sigma
