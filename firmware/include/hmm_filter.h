/*
 * Fixed-point forward filter for a four-regime hidden Markov model.
 *
 *   alpha_t(k) proportional to b_k(y_t) * sum_j a_jk alpha_{t-1}(j)
 *
 * The prediction A^T alpha runs as an int8 fully connected layer (hmm_fc_s8), which maps
 * to arm_fully_connected_s8 in CMSIS-NN when HMM_USE_CMSIS_NN is defined and to a portable
 * C99 reference with the same integer semantics otherwise. The emission term uses the
 * lifted field offsets (P x_hat) for the current sensor site. A fixed-point rescale keeps
 * the belief normalised so it never underflows; the largest entry is the current mode.
 *
 * Belief encoding: int8 q represents (q + 128) / 255.
 * Regimes: 0 calm, 1 alert, 2 high-stress, 3 panic.
 */
#ifndef HMM_FILTER_H
#define HMM_FILTER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define HMM_K 4
#define HMM_D 2

enum { HMM_CALM = 0, HMM_ALERT = 1, HMM_HIGH_STRESS = 2, HMM_PANIC = 3 };

enum {
    HMM_OK = 0,
    HMM_ERR_NULL = -1,
    HMM_ERR_SITE = -2,
    HMM_ERR_FC = -3
};

typedef struct {
    int8_t w[HMM_K][HMM_K];          /* row k: weights into regime k (A transposed) */
    int32_t bias[HMM_K];
    int32_t fc_multiplier;           /* Q31 */
    int32_t fc_shift;                /* positive: left shift */
    int32_t input_offset;
    int32_t output_offset;
    int32_t mu[HMM_K][HMM_D];        /* emission means, mm */
    int32_t inv_sigma_q16[HMM_K][HMM_D];
    int32_t lognorm_q16[HMM_K];      /* -sum_d log sigma_kd, Q16 */
    uint8_t offset_channel_mask;     /* channels that receive the lifted offset (H) */
    const int16_t *site_offset_mm;   /* lifted offsets P x_hat, one per site */
    uint16_t n_sites;
    int8_t init_belief[HMM_K];
} hmm_model_t;

typedef struct {
    const hmm_model_t *model;
    int8_t belief[HMM_K];
    uint32_t steps;
    uint32_t floor_hits;             /* predictions raised to the probability floor */
} hmm_state_t;

int hmm_init(hmm_state_t *st, const hmm_model_t *model);

/* One recursion step. y_mm: lateral offset and SDLP window in millimetres.
 * Returns the mode (0..3) or a negative HMM_ERR_* code; on error the belief is unchanged. */
int hmm_step(hmm_state_t *st, const int16_t y_mm[HMM_D], uint16_t site);

/* Replace the model (for example after periodic re-estimation) and keep the belief. */
int hmm_swap_model(hmm_state_t *st, const hmm_model_t *model);

int hmm_mode(const hmm_state_t *st);

/* exp(-delta) in Q15 for delta >= 0 in Q16, floored at 1. Exposed for tests. */
int32_t hmm_exp_neg_q15(int64_t delta_q16);

#ifdef __cplusplus
}
#endif

#endif /* HMM_FILTER_H */
