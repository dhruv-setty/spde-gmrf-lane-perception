#include "hmm_filter.h"

#include <stddef.h>

#include "hmm_fc_s8.h"

extern const uint16_t hmm_exp2_lut_q15[256];

#define LOG2E_Q16 94548
#define Z_CLAMP_Q16 ((int64_t)1 << 24)

int32_t hmm_exp_neg_q15(int64_t delta_q16)
{
    int64_t t, ip;
    int32_t frac, e;
    if (delta_q16 < 0) {
        delta_q16 = 0;
    }
    t = (delta_q16 * LOG2E_Q16) >> 16;
    ip = t >> 16;
    if (ip >= 15) {
        return 1;
    }
    frac = (int32_t)((t >> 8) & 0xFF);
    e = (int32_t)(hmm_exp2_lut_q15[frac] >> ip);
    return e < 1 ? 1 : e;
}

int hmm_init(hmm_state_t *st, const hmm_model_t *model)
{
    int k;
    if (!st || !model) {
        return HMM_ERR_NULL;
    }
    st->model = model;
    for (k = 0; k < HMM_K; ++k) {
        st->belief[k] = model->init_belief[k];
    }
    st->steps = 0;
    st->floor_hits = 0;
    return HMM_OK;
}

int hmm_swap_model(hmm_state_t *st, const hmm_model_t *model)
{
    if (!st || !model) {
        return HMM_ERR_NULL;
    }
    st->model = model;
    return HMM_OK;
}

int hmm_mode(const hmm_state_t *st)
{
    int k, best = 0;
    for (k = 1; k < HMM_K; ++k) {
        if (st->belief[k] > st->belief[best]) {
            best = k;
        }
    }
    return best;
}

int hmm_step(hmm_state_t *st, const int16_t y_mm[HMM_D], uint16_t site)
{
    const hmm_model_t *m;
    int8_t pred[HMM_K];
    int64_t ll[HMM_K], mx, sum = 0;
    int64_t u[HMM_K];
    int k, d;

    if (!st || !st->model || !y_mm) {
        return HMM_ERR_NULL;
    }
    m = st->model;
    if (site >= m->n_sites) {
        return HMM_ERR_SITE;
    }

    /* Prediction: A^T alpha_{t-1} as an int8 fully connected layer. */
    if (hmm_fc_s8(st->belief, &m->w[0][0], m->bias, HMM_K, HMM_K, m->input_offset, m->output_offset,
                  m->fc_multiplier, m->fc_shift, -128, 127, pred) != 0) {
        return HMM_ERR_FC;
    }

    /* Emission log-likelihoods in Q16, using the lifted offset for this site. */
    for (k = 0; k < HMM_K; ++k) {
        int64_t sq = 0;
        for (d = 0; d < HMM_D; ++d) {
            int64_t mean = m->mu[k][d];
            int64_t z;
            if ((m->offset_channel_mask >> d) & 1u) {
                mean += m->site_offset_mm[site];
            }
            z = ((int64_t)y_mm[d] - mean) * (int64_t)m->inv_sigma_q16[k][d];
            if (z > Z_CLAMP_Q16) z = Z_CLAMP_Q16;
            if (z < -Z_CLAMP_Q16) z = -Z_CLAMP_Q16;
            sq += (z * z) >> 16;
        }
        ll[k] = (int64_t)m->lognorm_q16[k] - (sq >> 1);
    }
    mx = ll[0];
    for (k = 1; k < HMM_K; ++k) {
        if (ll[k] > mx) mx = ll[k];
    }

    /* Update with floors on both factors, then rescale to sum 255 (in offset form). */
    for (k = 0; k < HMM_K; ++k) {
        int64_t pk = (int64_t)pred[k] + 128;
        if (pk < 1) {
            pk = 1;
            st->floor_hits++;
        }
        u[k] = pk * (int64_t)hmm_exp_neg_q15(mx - ll[k]);
        sum += u[k];
    }
    for (k = 0; k < HMM_K; ++k) {
        int64_t q = (255 * u[k] + sum / 2) / sum - 128;
        if (q < -128) q = -128;
        if (q > 127) q = 127;
        st->belief[k] = (int8_t)q;
    }
    st->steps++;
    return hmm_mode(st);
}
