/*
 * CMSIS-NN backend for the prediction step. Compiled only with -DHMM_USE_CMSIS_NN and the
 * CMSIS-NN include path on an Arm Cortex-M target. It has not been compiled or run in this
 * repository's CI; check the call against the pinned CMSIS-NN release before use, in
 * particular whether that release expects precomputed kernel sums in ctx.buf.
 */
#ifdef HMM_USE_CMSIS_NN
#include <string.h>

#include "arm_nnfunctions.h"
#include "hmm_fc_s8.h"
#include "hmm_filter.h"

/* Scratch for kernels that request it; HMM_K * HMM_K is far below any realistic need. */
#ifndef HMM_CMSIS_SCRATCH_BYTES
#define HMM_CMSIS_SCRATCH_BYTES 256
#endif
static int32_t hmm_cmsis_scratch[HMM_CMSIS_SCRATCH_BYTES / sizeof(int32_t)];

int hmm_fc_s8(const int8_t *in, const int8_t *w, const int32_t *bias,
              int32_t in_ch, int32_t out_ch,
              int32_t input_offset, int32_t output_offset,
              int32_t multiplier, int32_t shift,
              int32_t act_min, int32_t act_max, int8_t *out)
{
    cmsis_nn_context ctx;
    cmsis_nn_fc_params fc_params;
    cmsis_nn_per_tensor_quant_params quant_params;
    cmsis_nn_dims input_dims = {1, 1, 1, in_ch};
    cmsis_nn_dims filter_dims = {in_ch, 1, 1, out_ch};
    cmsis_nn_dims bias_dims = {1, 1, 1, out_ch};
    cmsis_nn_dims output_dims = {1, 1, 1, out_ch};
    const int32_t buf_size = arm_fully_connected_s8_get_buffer_size(&filter_dims);

    if (buf_size > (int32_t)sizeof(hmm_cmsis_scratch)) {
        return -1;
    }
    memset(hmm_cmsis_scratch, 0, sizeof(hmm_cmsis_scratch));
    ctx.buf = buf_size > 0 ? hmm_cmsis_scratch : NULL;
    ctx.size = buf_size;

    fc_params.input_offset = input_offset;
    fc_params.filter_offset = 0;
    fc_params.output_offset = output_offset;
    fc_params.activation.min = act_min;
    fc_params.activation.max = act_max;
    quant_params.multiplier = multiplier;
    quant_params.shift = shift;

    return arm_fully_connected_s8(&ctx, &fc_params, &quant_params, &input_dims, in, &filter_dims, w,
                                  &bias_dims, bias, &output_dims, out) == ARM_CMSIS_NN_SUCCESS ? 0 : -1;
}
#endif /* HMM_USE_CMSIS_NN */
