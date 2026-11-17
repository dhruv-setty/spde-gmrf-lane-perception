// GPU evaluation of the SPDE-GMRF marginal likelihood over a grid of theta = (kappa, tau).
//
// Input: the case file written by `python -m spdegmrf.export --gpu-case FILE` (layout in
// python/spdegmrf/export.py, write_gpu_case). Output: one CSV line per theta,
// "kappa,tau,loglik", to compare with spdegmrf.estimate.Estimator.loglik on the CPU.
//
//   Q(theta) = tau^2 (kappa^4 C + 2 kappa^2 G + G C^-1 G)    values on a fixed pattern
//   Q_p      = Q + P^T P / sigma^2                            same pattern (checked)
//   loglik   = R/2 (log|Q| - log|Q_p|) - mR/2 log(2 pi sigma^2) - 1/2 (y'y/sigma^2 - b'mu)
//
// cuSPARSE forms b = P^T Y / sigma^2; cuDSS (which replaces the deprecated sparse cuSOLVER
// module) factorises Q and Q_p. The symbolic analysis of each runs once; every theta after
// that is a value update and a refactorisation. log|A| = 2 sum log L_ii is read from the
// factor diagonal (CUDSS_DATA_DIAG); confirm its meaning for SPD matrices against the cuDSS
// release in use. This file has not been compiled or run in this repository's CI.
#include <cuda_runtime.h>
#include <cudss.h>
#include <cusparse.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#define CK_CUDA(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) throw std::runtime_error(std::string(#x) + ": " + cudaGetErrorString(e_)); } while (0)
#define CK_SP(x) do { if ((x) != CUSPARSE_STATUS_SUCCESS) throw std::runtime_error(#x); } while (0)
#define CK_DSS(x) do { if ((x) != CUDSS_STATUS_SUCCESS) throw std::runtime_error(#x); } while (0)

struct Case {
    int n = 0, nnz = 0, m = 0, p_nnz = 0, R = 0;
    double sigma = 0.0;
    std::vector<int32_t> q_ptr, q_idx, p_ptr, p_idx;
    std::vector<double> c, g, k, p_val, y;  // y is m x R, column-major
};

template <class T> static void read_vec(std::ifstream &f, std::vector<T> &v, size_t count) {
    v.resize(count);
    f.read(reinterpret_cast<char *>(v.data()), static_cast<std::streamsize>(count * sizeof(T)));
}

static Case read_case(const char *path) {
    std::ifstream f(path, std::ios::binary);
    char magic[8];
    f.read(magic, 8);
    if (!f || std::memcmp(magic, "SPDECASE", 8) != 0) throw std::runtime_error("not a SPDECASE file");
    Case c;
    int32_t hdr[5];
    f.read(reinterpret_cast<char *>(hdr), sizeof hdr);
    f.read(reinterpret_cast<char *>(&c.sigma), sizeof c.sigma);
    c.n = hdr[0]; c.nnz = hdr[1]; c.m = hdr[2]; c.p_nnz = hdr[3]; c.R = hdr[4];
    read_vec(f, c.q_ptr, c.n + 1); read_vec(f, c.q_idx, c.nnz);
    read_vec(f, c.c, c.nnz); read_vec(f, c.g, c.nnz); read_vec(f, c.k, c.nnz);
    read_vec(f, c.p_ptr, c.m + 1); read_vec(f, c.p_idx, c.p_nnz); read_vec(f, c.p_val, c.p_nnz);
    read_vec(f, c.y, static_cast<size_t>(c.m) * c.R);
    if (!f) throw std::runtime_error("truncated case file");
    return c;
}

// P^T P / sigma^2 on the pattern of Q (host, once). Fails if a coupling falls outside it.
static std::vector<double> ptp_on_pattern(const Case &c) {
    std::vector<double> out(c.nnz, 0.0);
    for (int s = 0; s < c.m; ++s)
        for (int a = c.p_ptr[s]; a < c.p_ptr[s + 1]; ++a)
            for (int b = c.p_ptr[s]; b < c.p_ptr[s + 1]; ++b) {
                int col = c.p_idx[a], row = c.p_idx[b], pos = -1;
                for (int t = c.q_ptr[col]; t < c.q_ptr[col + 1]; ++t)
                    if (c.q_idx[t] == row) { pos = t; break; }
                if (pos < 0) throw std::runtime_error("P^T P outside the pattern of Q");
                out[pos] += c.p_val[a] * c.p_val[b] / (c.sigma * c.sigma);
            }
    return out;
}

__global__ void build_q(int nnz, double kappa, double tau, const double *c, const double *g, const double *k,
                        const double *ptp, double *q, double *qp) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nnz) return;
    double k2 = kappa * kappa, t2 = tau * tau;
    double v = t2 * (k2 * k2 * c[i] + 2.0 * k2 * g[i] + k[i]);
    q[i] = v;
    qp[i] = v + ptp[i];
}

struct Solver {  // one cuDSS factorisation with a reused analysis
    cudssMatrix_t A{}, X{}, B{};
    cudssData_t data{};
    bool analysed = false, factored = false;
};

static double logdet(cudssHandle_t h, Solver &s, int n) {
    std::vector<double> d(n);
    double *d_diag = nullptr;
    size_t written = 0;
    CK_CUDA(cudaMalloc(&d_diag, n * sizeof(double)));
    CK_DSS(cudssDataGet(h, s.data, CUDSS_DATA_DIAG, d_diag, n * sizeof(double), &written));
    CK_CUDA(cudaMemcpy(d.data(), d_diag, n * sizeof(double), cudaMemcpyDeviceToHost));
    CK_CUDA(cudaFree(d_diag));
    double acc = 0.0;
    for (double v : d) acc += std::log(v);
    return 2.0 * acc;
}

static void factor(cudssHandle_t h, cudssConfig_t cfg, Solver &s) {
    if (!s.analysed) { CK_DSS(cudssExecute(h, CUDSS_PHASE_ANALYSIS, cfg, s.data, s.A, s.X, s.B)); s.analysed = true; }
    CK_DSS(cudssExecute(h, s.factored ? CUDSS_PHASE_REFACTORIZATION : CUDSS_PHASE_FACTORIZATION,
                        cfg, s.data, s.A, s.X, s.B));
    s.factored = true;
}

int main(int argc, char **argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s CASE [kappa tau]...\n", argv[0]); return 2; }
    try {
        Case c = read_case(argv[1]);
        std::vector<std::pair<double, double>> grid;
        for (int a = 2; a + 1 < argc; a += 2) grid.emplace_back(std::atof(argv[a]), std::atof(argv[a + 1]));
        if (grid.empty()) grid = {{4.0, 0.0015}, {4.7, 0.0015}, {5.5, 0.0015}};
        const int n = c.n, m = c.m, R = c.R, nnz = c.nnz;
        std::vector<double> ptp = ptp_on_pattern(c);

        int32_t *d_qptr, *d_qidx, *d_pptr, *d_pidx;
        double *d_c, *d_g, *d_k, *d_ptp, *d_q, *d_qp, *d_pval, *d_y, *d_b, *d_x;
        CK_CUDA(cudaMalloc(&d_qptr, (n + 1) * sizeof(int32_t)));
        CK_CUDA(cudaMalloc(&d_qidx, nnz * sizeof(int32_t)));
        CK_CUDA(cudaMalloc(&d_pptr, (m + 1) * sizeof(int32_t)));
        CK_CUDA(cudaMalloc(&d_pidx, c.p_nnz * sizeof(int32_t)));
        for (double **p : {&d_c, &d_g, &d_k, &d_ptp, &d_q, &d_qp}) CK_CUDA(cudaMalloc(p, nnz * sizeof(double)));
        CK_CUDA(cudaMalloc(&d_pval, c.p_nnz * sizeof(double)));
        CK_CUDA(cudaMalloc(&d_y, static_cast<size_t>(m) * R * sizeof(double)));
        CK_CUDA(cudaMalloc(&d_b, static_cast<size_t>(n) * R * sizeof(double)));
        CK_CUDA(cudaMalloc(&d_x, static_cast<size_t>(n) * R * sizeof(double)));
        CK_CUDA(cudaMemcpy(d_qptr, c.q_ptr.data(), (n + 1) * sizeof(int32_t), cudaMemcpyHostToDevice));
        CK_CUDA(cudaMemcpy(d_qidx, c.q_idx.data(), nnz * sizeof(int32_t), cudaMemcpyHostToDevice));
        CK_CUDA(cudaMemcpy(d_pptr, c.p_ptr.data(), (m + 1) * sizeof(int32_t), cudaMemcpyHostToDevice));
        CK_CUDA(cudaMemcpy(d_pidx, c.p_idx.data(), c.p_nnz * sizeof(int32_t), cudaMemcpyHostToDevice));
        CK_CUDA(cudaMemcpy(d_c, c.c.data(), nnz * sizeof(double), cudaMemcpyHostToDevice));
        CK_CUDA(cudaMemcpy(d_g, c.g.data(), nnz * sizeof(double), cudaMemcpyHostToDevice));
        CK_CUDA(cudaMemcpy(d_k, c.k.data(), nnz * sizeof(double), cudaMemcpyHostToDevice));
        CK_CUDA(cudaMemcpy(d_ptp, ptp.data(), nnz * sizeof(double), cudaMemcpyHostToDevice));
        CK_CUDA(cudaMemcpy(d_pval, c.p_val.data(), c.p_nnz * sizeof(double), cudaMemcpyHostToDevice));
        CK_CUDA(cudaMemcpy(d_y, c.y.data(), c.y.size() * sizeof(double), cudaMemcpyHostToDevice));

        // b = P^T Y / sigma^2 with cuSPARSE SpMM (independent of theta, computed once).
        cusparseHandle_t sp;
        CK_SP(cusparseCreate(&sp));
        cusparseSpMatDescr_t P;
        cusparseDnMatDescr_t Yd, Bd;
        CK_SP(cusparseCreateCsr(&P, m, n, c.p_nnz, d_pptr, d_pidx, d_pval, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
        CK_SP(cusparseCreateDnMat(&Yd, m, R, m, d_y, CUDA_R_64F, CUSPARSE_ORDER_COL));
        CK_SP(cusparseCreateDnMat(&Bd, n, R, n, d_b, CUDA_R_64F, CUSPARSE_ORDER_COL));
        const double alpha = 1.0 / (c.sigma * c.sigma), beta = 0.0;
        size_t ws = 0;
        void *d_ws = nullptr;
        CK_SP(cusparseSpMM_bufferSize(sp, CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, P,
                                      Yd, &beta, Bd, CUDA_R_64F, CUSPARSE_SPMM_ALG_DEFAULT, &ws));
        CK_CUDA(cudaMalloc(&d_ws, ws ? ws : 1));
        CK_SP(cusparseSpMM(sp, CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, P, Yd, &beta,
                           Bd, CUDA_R_64F, CUSPARSE_SPMM_ALG_DEFAULT, d_ws));
        std::vector<double> b(static_cast<size_t>(n) * R);
        CK_CUDA(cudaMemcpy(b.data(), d_b, b.size() * sizeof(double), cudaMemcpyDeviceToHost));
        double yy = 0.0;
        for (double v : c.y) yy += v * v;

        cudssHandle_t h;
        cudssConfig_t cfg;
        CK_DSS(cudssCreate(&h));
        CK_DSS(cudssConfigCreate(&cfg));
        Solver sq, sqp;
        for (Solver *s : {&sq, &sqp}) {
            CK_DSS(cudssDataCreate(h, &s->data));
            CK_DSS(cudssMatrixCreateDn(&s->X, n, R, n, d_x, CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR));
            CK_DSS(cudssMatrixCreateDn(&s->B, n, R, n, d_b, CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR));
        }
        CK_DSS(cudssMatrixCreateCsr(&sq.A, n, n, nnz, d_qptr, nullptr, d_qidx, d_q, CUDA_R_32I, CUDA_R_64F,
                                    CUDSS_MTYPE_SPD, CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO));
        CK_DSS(cudssMatrixCreateCsr(&sqp.A, n, n, nnz, d_qptr, nullptr, d_qidx, d_qp, CUDA_R_32I, CUDA_R_64F,
                                    CUDSS_MTYPE_SPD, CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO));

        std::printf("kappa,tau,loglik\n");
        std::vector<double> mu(static_cast<size_t>(n) * R);
        for (auto [kappa, tau] : grid) {
            build_q<<<(nnz + 255) / 256, 256>>>(nnz, kappa, tau, d_c, d_g, d_k, d_ptp, d_q, d_qp);
            CK_CUDA(cudaGetLastError());
            factor(h, cfg, sq);
            factor(h, cfg, sqp);
            CK_DSS(cudssExecute(h, CUDSS_PHASE_SOLVE, cfg, sqp.data, sqp.A, sqp.X, sqp.B));
            CK_CUDA(cudaMemcpy(mu.data(), d_x, mu.size() * sizeof(double), cudaMemcpyDeviceToHost));
            double bmu = 0.0;
            for (size_t i = 0; i < mu.size(); ++i) bmu += b[i] * mu[i];
            const double ll = 0.5 * R * (logdet(h, sq, n) - logdet(h, sqp, n)) -
                              0.5 * m * R * std::log(2.0 * M_PI * c.sigma * c.sigma) -
                              0.5 * (yy / (c.sigma * c.sigma) - bmu);
            std::printf("%.10g,%.10g,%.12g\n", kappa, tau, ll);
        }
        // Process exit releases device memory and library handles.
    } catch (const std::exception &e) {
        std::fprintf(stderr, "spde_gpu: %s\n", e.what());
        return 1;
    }
    return 0;
}
