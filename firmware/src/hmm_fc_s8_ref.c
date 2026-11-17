#include "hmm_fc_s8.h"

static int32_t doubling_high_mult_no_sat(int32_t m1, int32_t m2)
{
    int64_t mult = ((int64_t)1 << 30) + (int64_t)m1 * (int64_t)m2;
    return (int32_t)(mult >> 31);
}

static int32_t divide_by_power_of_two(int32_t dividend, int32_t exponent)
{
    const int32_t remainder_mask = (int32_t)((1u << exponent) - 1u);
    const int32_t remainder = remainder_mask & dividend;
    int32_t result = dividend >> exponent;
    int32_t threshold = remainder_mask >> 1;
    if (result < 0) {
        threshold++;
    }
    if (remainder > threshold) {
        result++;
    }
    return result;
}

int32_t hmm_requantize(int32_t val, int32_t multiplier, int32_t shift)
{
    const int32_t left = shift > 0 ? shift : 0;
    const int32_t right = shift > 0 ? 0 : -shift;
    const int32_t scaled = (int32_t)((uint32_t)val << left);
    return divide_by_power_of_two(doubling_high_mult_no_sat(scaled, multiplier), right);
}

int hmm_fc_s8_ref(const int8_t *in, const int8_t *w, const int32_t *bias,
                  int32_t in_ch, int32_t out_ch,
                  int32_t input_offset, int32_t output_offset,
                  int32_t multiplier, int32_t shift,
                  int32_t act_min, int32_t act_max, int8_t *out)
{
    int32_t k, j;
    if (!in || !w || !out || in_ch <= 0 || out_ch <= 0) {
        return -1;
    }
    for (k = 0; k < out_ch; ++k) {
        int32_t acc = bias ? bias[k] : 0;
        for (j = 0; j < in_ch; ++j) {
            acc += ((int32_t)in[j] + input_offset) * (int32_t)w[k * in_ch + j];
        }
        acc = hmm_requantize(acc, multiplier, shift) + output_offset;
        if (acc < act_min) acc = act_min;
        if (acc > act_max) acc = act_max;
        out[k] = (int8_t)acc;
    }
    return 0;
}

#ifndef HMM_USE_CMSIS_NN
int hmm_fc_s8(const int8_t *in, const int8_t *w, const int32_t *bias,
              int32_t in_ch, int32_t out_ch,
              int32_t input_offset, int32_t output_offset,
              int32_t multiplier, int32_t shift,
              int32_t act_min, int32_t act_max, int8_t *out)
{
    return hmm_fc_s8_ref(in, w, bias, in_ch, out_ch, input_offset, output_offset,
                         multiplier, shift, act_min, act_max, out);
}
#endif
