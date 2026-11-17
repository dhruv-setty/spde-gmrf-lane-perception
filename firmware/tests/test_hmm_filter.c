/* Host unit tests for the fixed-point forward filter. Build and run with `make test`. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "hmm_fc_s8.h"
#include "hmm_filter.h"
#include "hmm_model_generated.h"
#include "hmm_vectors_generated.h"

static int failures = 0;
#define CHECK(cond)                                                                 \
    do {                                                                            \
        if (!(cond)) {                                                              \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);          \
            failures++;                                                             \
        }                                                                           \
    } while (0)

static void test_requantize(void)
{
    /* multiplier 2^30 (0.5 in Q31) with shift 0 halves with rounding half away from zero. */
    CHECK(hmm_requantize(10, 1 << 30, 0) == 5);
    CHECK(hmm_requantize(11, 1 << 30, 0) == 6);
    CHECK(hmm_requantize(-11, 1 << 30, 0) == -5);
    /* shift -1 halves again: 100 * 0.5 / 2 = 25. */
    CHECK(hmm_requantize(100, 1 << 30, -1) == 25);
    /* positive shift multiplies first: 3 * 4 * 0.5 = 6. */
    CHECK(hmm_requantize(3, 1 << 30, 2) == 6);
}

static void test_fc_ref(void)
{
    const int8_t in[2] = {-128, 127};
    const int8_t w[4] = {1, 2, 3, -4};
    const int32_t bias[2] = {0, 10};
    int8_t out[2];
    /* inputs +128 -> {0, 255}; acc = {510, -1010}; multiplier 0.5 -> {255, -500}; clamp. */
    CHECK(hmm_fc_s8_ref(in, w, bias, 2, 2, 128, 0, 1 << 30, 0, -128, 127, out) == 0);
    CHECK(out[0] == 127);
    CHECK(out[1] == -128);
    CHECK(hmm_fc_s8_ref(in, w, bias, 2, 2, 128, -100, 1 << 30, -2, -128, 127, out) == 0);
    /* 510 * 0.5 / 4 = 63.75 -> 64; -100 -> -36. (-1000) * 0.5 / 4 = -125 -> -225 -> clamp -128. */
    CHECK(out[0] == -36);
    CHECK(out[1] == -128);
}

static void test_exp(void)
{
    int64_t d;
    int32_t prev = 1 << 15;
    CHECK(hmm_exp_neg_q15(0) == 32768);
    for (d = 0; d < ((int64_t)40 << 16); d += 997) {
        int32_t e = hmm_exp_neg_q15(d);
        CHECK(e >= 1 && e <= prev);
        prev = e;
    }
    CHECK(hmm_exp_neg_q15((int64_t)1000 << 16) == 1);
}

static void test_model_shape(void)
{
    /* The direct calm -> panic path a_14 survives quantisation. */
    CHECK(hmm_gen_model.w[HMM_PANIC][HMM_CALM] > 0);
    CHECK(hmm_gen_model.n_sites == HMM_GEN_N_SITES);
}

static void test_vectors(void)
{
    hmm_state_t st;
    int t, k;
    int exact = 1, agree_float = 0, int8_correct = 0, float_correct = 0;
    CHECK(hmm_init(&st, &hmm_gen_model) == HMM_OK);
    for (t = 0; t < HMM_VEC_T; ++t) {
        int mode = hmm_step(&st, &hmm_vec_obs_mm[2 * t], hmm_vec_site[t]);
        CHECK(mode >= 0);
        for (k = 0; k < HMM_K; ++k) {
            if (st.belief[k] != hmm_vec_expected_belief[HMM_K * t + k]) {
                exact = 0;
            }
        }
        if (mode != hmm_vec_expected_mode[t]) exact = 0;
        agree_float += (mode == hmm_vec_float_mode[t]);
        int8_correct += (mode == hmm_vec_true_regime[t]);
        float_correct += (hmm_vec_float_mode[t] == hmm_vec_true_regime[t]);
    }
    CHECK(exact);
    printf("vectors: T=%d bit_exact_vs_python=%s int8_vs_float_argmax_agreement=%.4f "
           "int8_accuracy=%.4f float_accuracy=%.4f floor_hits=%u (simulated data)\n",
           HMM_VEC_T, exact ? "yes" : "no", (double)agree_float / HMM_VEC_T,
           (double)int8_correct / HMM_VEC_T, (double)float_correct / HMM_VEC_T, (unsigned)st.floor_hits);
    CHECK((double)agree_float / HMM_VEC_T >= 0.97);
}

static void test_no_underflow(void)
{
    hmm_state_t st;
    int t, k;
    const int16_t extreme[2] = {32767, 32767};
    const int16_t other[2] = {-32768, 0};
    hmm_init(&st, &hmm_gen_model);
    for (t = 0; t < 200; ++t) {
        int sum = 0;
        int mode = hmm_step(&st, (t & 1) ? extreme : other, (uint16_t)(t % HMM_GEN_N_SITES));
        CHECK(mode >= 0 && mode < HMM_K);
        for (k = 0; k < HMM_K; ++k) sum += st.belief[k] + 128;
        /* Rounding of four entries moves the sum by at most two from 255; never all zero. */
        CHECK(sum >= 253 && sum <= 257);
    }
}

static void test_bad_site(void)
{
    hmm_state_t st;
    int8_t before[HMM_K];
    const int16_t y[2] = {0, 120};
    hmm_init(&st, &hmm_gen_model);
    memcpy(before, st.belief, sizeof before);
    CHECK(hmm_step(&st, y, HMM_GEN_N_SITES) == HMM_ERR_SITE);
    CHECK(memcmp(before, st.belief, sizeof before) == 0);
    CHECK(hmm_step(NULL, y, 0) == HMM_ERR_NULL);
}

static void test_calm_to_panic(void)
{
    /* Start calm, then a sustained panic-like signal must move the mode to panic. */
    hmm_state_t st;
    int t, mode = -1;
    const int16_t calm[2] = {0, 120};
    const int16_t panic[2] = {80, 430};
    hmm_init(&st, &hmm_gen_model);
    for (t = 0; t < 50; ++t) mode = hmm_step(&st, calm, 0);
    CHECK(mode == HMM_CALM);
    for (t = 0; t < 10; ++t) mode = hmm_step(&st, panic, 0);
    CHECK(mode == HMM_PANIC);
}

int main(void)
{
    test_requantize();
    test_fc_ref();
    test_exp();
    test_model_shape();
    test_vectors();
    test_no_underflow();
    test_bad_site();
    test_calm_to_panic();
    if (failures) {
        fprintf(stderr, "%d check(s) failed\n", failures);
        return EXIT_FAILURE;
    }
    printf("firmware tests: all passed\n");
    return EXIT_SUCCESS;
}
