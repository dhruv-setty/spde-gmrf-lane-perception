/*
 * int8 fully connected layer with arm_fully_connected_s8 semantics (batch 1):
 *
 *   out[k] = clamp(requantize(bias[k] + sum_j (in[j] + input_offset) * w[k * in_ch + j],
 *                             multiplier, shift) + output_offset, act_min, act_max)
 *
 * requantize follows the CMSIS-NN default path (no CMSIS_NN_USE_SINGLE_ROUNDING):
 * a rounding doubling high multiply followed by a rounding divide by a power of two.
 */
#ifndef HMM_FC_S8_H
#define HMM_FC_S8_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int hmm_fc_s8(const int8_t *in, const int8_t *w, const int32_t *bias,
              int32_t in_ch, int32_t out_ch,
              int32_t input_offset, int32_t output_offset,
              int32_t multiplier, int32_t shift,
              int32_t act_min, int32_t act_max, int8_t *out);

/* Portable reference, always compiled so the host tests can compare backends. */
int hmm_fc_s8_ref(const int8_t *in, const int8_t *w, const int32_t *bias,
                  int32_t in_ch, int32_t out_ch,
                  int32_t input_offset, int32_t output_offset,
                  int32_t multiplier, int32_t shift,
                  int32_t act_min, int32_t act_max, int8_t *out);

int32_t hmm_requantize(int32_t val, int32_t multiplier, int32_t shift);

#ifdef __cplusplus
}
#endif

#endif /* HMM_FC_S8_H */
