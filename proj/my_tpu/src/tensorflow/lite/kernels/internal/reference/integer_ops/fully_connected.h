/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_FULLY_CONNECTED_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_FULLY_CONNECTED_H_

#include <algorithm>
#include <stdio.h>
#include "tensorflow/lite/kernels/internal/common.h"
#include "cfu.h"
#include <iostream>


namespace tflite {
namespace reference_integer_ops {

// For per-channel functions, since it is defined in quantization spec that
// weights are symmetric
// (https://www.tensorflow.org/lite/performance/quantization_spec#symmetric_vs_asymmetric),
// zero_point (params.weights_offset) is always 0.
// However, for per-tensor functions, params.weights_offset is still applied for
// backward compatibility.

inline void FullyConnectedPerChannel(
    const FullyConnectedParams& params, const int32_t* output_multiplier,
    const int* output_shift, const RuntimeShape& input_shape,
    const int8_t* input_data, const RuntimeShape& filter_shape,
    const int8_t* filter_data, const RuntimeShape& bias_shape,
    const int32_t* bias_data, const RuntimeShape& output_shape,
    int8_t* output_data) {
  const int32_t input_offset = params.input_offset;
  const int32_t output_offset = params.output_offset;
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;
  TFLITE_DCHECK_GE(filter_shape.DimensionsCount(), 2);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 2);

  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  const int filter_dim_count = filter_shape.DimensionsCount();
  const int batches = output_shape.Dims(0);
  const int output_depth = output_shape.Dims(1);
  TFLITE_DCHECK_LE(output_depth, filter_shape.Dims(filter_dim_count - 2));
  const int accum_depth = filter_shape.Dims(filter_dim_count - 1);
  for (int b = 0; b < batches; ++b) {
    for (int out_c = 0; out_c < output_depth; ++out_c) {
      int32_t acc = 0;
      for (int d = 0; d < accum_depth; ++d) {
        int32_t input_val = input_data[b * accum_depth + d];
        int32_t filter_val = filter_data[out_c * accum_depth + d];
        acc += filter_val * (input_val + input_offset);
      }
      if (bias_data) {
        acc += bias_data[out_c];
      }
      acc = MultiplyByQuantizedMultiplier(acc, output_multiplier[out_c],
                                          output_shift[out_c]);
      acc += output_offset;
      acc = std::max(acc, output_activation_min);
      acc = std::min(acc, output_activation_max);
      output_data[out_c + output_depth * b] = static_cast<int8_t>(acc);
    }
  }
}

template <typename AccumScalar>
inline void FullyConnectedPerChannel(
    const FullyConnectedParams& params, const int32_t* output_multiplier,
    const int* output_shift, const RuntimeShape& input_shape,
    const int16_t* input_data, const RuntimeShape& filter_shape,
    const int8_t* filter_data, const RuntimeShape& bias_shape,
    const AccumScalar* bias_data, const RuntimeShape& output_shape,
    int16_t* output_data) {
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;
  TFLITE_DCHECK_GE(filter_shape.DimensionsCount(), 2);
  TFLITE_DCHECK_GE(output_shape.DimensionsCount(), 1);

  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  const int filter_dim_count = filter_shape.DimensionsCount();
  const int output_dim_count = output_shape.DimensionsCount();
  const int batches = FlatSizeSkipDim(output_shape, output_dim_count - 1);
  const int output_depth = output_shape.Dims(output_dim_count - 1);
  TFLITE_DCHECK_LE(output_depth, filter_shape.Dims(filter_dim_count - 2));
  const int accum_depth = filter_shape.Dims(filter_dim_count - 1);
  for (int b = 0; b < batches; ++b) {
    for (int out_c = 0; out_c < output_depth; ++out_c) {
      AccumScalar acc = 0;
      for (int d = 0; d < accum_depth; ++d) {
        int32_t input_val = input_data[b * accum_depth + d];
        int32_t filter_val = filter_data[out_c * accum_depth + d];
        acc += filter_val * input_val;
      }
      if (bias_data) {
        acc += bias_data[out_c];
      }
      int32_t acc_scaled = MultiplyByQuantizedMultiplier(
          acc, output_multiplier[out_c], output_shift[out_c]);
      acc_scaled = std::max(acc_scaled, output_activation_min);
      acc_scaled = std::min(acc_scaled, output_activation_max);
      output_data[out_c + output_depth * b] = static_cast<int16_t>(acc_scaled);
    }
  }
}

inline void FullyConnected(
    const FullyConnectedParams& params, const RuntimeShape& input_shape,
    const int8_t* input_data, const RuntimeShape& filter_shape,
    const int8_t* filter_data, const RuntimeShape& bias_shape,
    const int32_t* bias_data, const RuntimeShape& output_shape,
    int8_t* output_data) {
  const int32_t input_offset = params.input_offset;
  //const int32_t filter_offset = params.weights_offset;
  const int32_t output_offset = params.output_offset;
  const int32_t output_multiplier = params.output_multiplier;
  const int output_shift = params.output_shift;
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;
  TFLITE_DCHECK_GE(filter_shape.DimensionsCount(), 2);
  TFLITE_DCHECK_GE(output_shape.DimensionsCount(), 1);

  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  const int filter_dim_count = filter_shape.DimensionsCount();
  const int output_dim_count = output_shape.DimensionsCount();
  //const int batches = FlatSizeSkipDim(output_shape, output_dim_count - 1);
  const int output_depth = output_shape.Dims(output_dim_count - 1);
  TFLITE_DCHECK_LE(output_depth, filter_shape.Dims(filter_dim_count - 2));
  const int accum_depth = filter_shape.Dims(filter_dim_count - 1);


  //printf("batch: %d, accum_depth:%d, output_depth:%d\n", batches, accum_depth, output_depth);
  /*
  printf("input_offset: %ld, filter_offset:%ld\n", input_offset, filter_offset);
  uint32_t input = ((uint8_t)(input_data[0]) << 24u ) |
                         ((uint8_t)(input_data[1]) << 16u ) |
                         ((uint8_t)(input_data[2]) << 8u  ) |
                         ((uint8_t)(input_data[3]) << 0u  ) ;
  printf("Input: %u\n %u\n %u\n %u\n",(uint8_t)(input_data[0]) << 24u ,  
                                          (uint8_t)(input_data[1]) << 16u , 
                                          (uint8_t)(input_data[2]) << 8u,
                                          (uint8_t)(input_data[3]) << 0u );
  printf("Send data %d, %d, %d, %d into CFU\n", input_data[0], input_data[1], input_data[2], input_data[3]);
  */
  //uint32_t cfulll = cfu_op1(/* funct7= */ 1, /* in0= */ 0, /* in1= */ input); 
  //printf("%lu\n", cfulll);
  //cfulll = cfu_op1(/* funct7= */ 0, /* in0= */ 0, /* in1= */ 5678);
  //printf("%lu\n", cfulll);

  


  /*
  for (int b = 0; b < batches; ++b) {
    for (int out_c = 0; out_c < output_depth; ++out_c) {
      int32_t acc  = 0;
      for (int d = 0; d < accum_depth; ++d) {
        int32_t input_val = input_data[b * accum_depth + d];
        int32_t filter_val = filter_data[out_c * accum_depth + d];
        //acc += (filter_val + filter_offset) * (input_val + input_offset);
        acc += (filter_val * input_val)  + (input_offset * filter_val);  
               // since filter_offset is zero, we can removed the corresponding calcultation to simplified the calculation
               //(filter_offset * input_val) +  // TPU 
               //(filter_offset * input_offset); // Precalculate
      }
      if (bias_data) {
        acc += bias_data[out_c];
      }
      acc = MultiplyByQuantizedMultiplier(acc, output_multiplier, output_shift);
      acc += output_offset;
      acc = std::max(acc, output_activation_min);
      acc = std::min(acc, output_activation_max);
      output_data[out_c + output_depth * b] = static_cast<int8_t>(acc);
    }
  }
  */

  /*
  //for (int b = 0; b < batches; ++b) {
    for (int out_c = 0; out_c < output_depth; out_c +=4 ) {
      int32_t acc[4]  = {0};
      for (int d = 0; d < accum_depth; ++d) {

        for (int i = 0 ; i < 4 && (out_c + i < output_depth); i++){
          int32_t input_val = input_data[0 * accum_depth + d];
          int32_t filter_val = filter_data[(out_c + i) * accum_depth + d];
          //acc += (filter_val + filter_offset) * (input_val + input_offset);
          acc[i] += (filter_val * input_val)  + (input_offset * filter_val);  
                // since filter_offset is zero, we can removed the corresponding calcultation to simplified the calculation
                //(filter_offset * input_val) +  // TPU 
                //(filter_offset * input_offset); // Precalculate
        } 
      }
      for(int i = 0 ; i < 4 && (out_c + i < output_depth); i++){
        if (bias_data) {
          acc[i] += bias_data[out_c + i];
        }
        acc[i] = MultiplyByQuantizedMultiplier(acc[i], output_multiplier, output_shift);
        acc[i] += output_offset;
        acc[i] = std::max(acc[i], output_activation_min);
        acc[i] = std::min(acc[i], output_activation_max);
        output_data[out_c + output_depth * 0 + i] = static_cast<int8_t>(acc[i]);
      }
    }
  //}
  */
  
  for (int out_c = 0; out_c < output_depth; out_c+=4){
    int32_t acc[4]  = {0};
    //int32_t acc2[4]  = {0};
    for (int d = 0; d < accum_depth; ++d){
        // input, load 4 input values {input_val[d], 0, 0, 0} to buffer A;
        // filter, load 4 filter values {filter_val[(out_c + 0) * accum_depth + d], 
        //                               filter_val[(out_c + 1) * accum_depth + d], 
        //                               filter_val[(out_c + 2) * accum_depth + d], 
        //                               filter_val[(out_c + 3) * accum_depth + d]} to buffer B,

        uint32_t value_to_buffer_a = ((uint8_t)(input_data[d]) << 24u );
        
        cfu_op0(/* funct7= */ 0, /* in0= */ d, /* in1= */ value_to_buffer_a);

        uint32_t value_to_buffer_b = ((uint8_t)(filter_data[(out_c + 0) * accum_depth + d]) << 24u ) |
                                     ((uint8_t)(filter_data[(out_c + 1) * accum_depth + d]) << 16u ) |
                                     ((uint8_t)(filter_data[(out_c + 2) * accum_depth + d]) << 8u  ) |
                                     ((uint8_t)(filter_data[(out_c + 3) * accum_depth + d]) << 0u  ) ;
        cfu_op0(/* funct7= */ 1, /* in0= */ d, /* in1= */ value_to_buffer_b);
        
        acc[0] += (int32_t)(filter_data[(out_c + 0) * accum_depth + d]);
        acc[1] += (int32_t)(filter_data[(out_c + 1) * accum_depth + d]);
        acc[2] += (int32_t)(filter_data[(out_c + 2) * accum_depth + d]);
        acc[3] += (int32_t)(filter_data[(out_c + 3) * accum_depth + d]);



        //acc2[0] += (int32_t)(filter_data[(out_c + 0) * accum_depth + d]) * input_data[d] ;
        //acc2[1] += (int32_t)(filter_data[(out_c + 1) * accum_depth + d]) * input_data[d];
        //acc2[2] += (int32_t)(filter_data[(out_c + 2) * accum_depth + d]) * input_data[d];
        //acc2[3] += (int32_t)(filter_data[(out_c + 3) * accum_depth + d]  * input_data[d]);
    }
    //printf("%ld, %ld, %ld, %ld\n", acc2[0], acc2[1], acc2[2], acc2[3]);
    cfu_op0(/* funct7= */ 2, /* in0= */ accum_depth, /* in1= */ 4);
    // calculate
    //int32_t results[4];
    //results[0] = cfu_op0(/* funct7= */ 3, /* in0= */ 0, /* in1= */ 0);
    //results[1] = cfu_op0(/* funct7= */ 3, /* in0= */ 0, /* in1= */ 1);
    //results[2] = cfu_op0(/* funct7= */ 3, /* in0= */ 0, /* in1= */ 2);
    //results[3] = cfu_op0(/* funct7= */ 3, /* in0= */ 0, /* in1= */ 3);
    //printf("%ld, %ld, %ld, %ld\n", results[0], results[1], results[2], results[3]);
    
    for (int i = 0; i < 4 && (out_c + i < output_depth); i++){
      //int32_t acc = (fetch result from CFU);
      int32_t cfu_query_result = cfu_op0(/* funct7= */ 3, /* in0= */ 0, /* in1= */ i);
      cfu_query_result += acc[i] * input_offset;
      // add bias
      if (bias_data) {
          cfu_query_result += bias_data[out_c + i];
      }
      cfu_query_result = MultiplyByQuantizedMultiplier(cfu_query_result, output_multiplier, output_shift);
      cfu_query_result += output_offset;
      cfu_query_result = std::max(cfu_query_result, output_activation_min);
      cfu_query_result = std::min(cfu_query_result, output_activation_max);
      output_data[out_c + output_depth * 0 + i] = static_cast<int8_t>(cfu_query_result);
      // quantization (multiply the scaling factor)
      // quantization (offset)
      // quantization (clamp)
      // store the result
    }
  }

}

template <typename AccumScalar>
inline void FullyConnected(
    const FullyConnectedParams& params, const RuntimeShape& input_shape,
    const int16_t* input_data, const RuntimeShape& filter_shape,
    const int8_t* filter_data, const RuntimeShape& bias_shape,
    const AccumScalar* bias_data, const RuntimeShape& output_shape,
    int16_t* output_data) {
  const int32_t filter_offset = params.weights_offset;
  const int32_t output_multiplier = params.output_multiplier;
  const int output_shift = params.output_shift;
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;
  TFLITE_DCHECK_GE(filter_shape.DimensionsCount(), 2);
  TFLITE_DCHECK_GE(output_shape.DimensionsCount(), 1);

  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  const int filter_dim_count = filter_shape.DimensionsCount();
  const int output_dim_count = output_shape.DimensionsCount();
  const int batches = FlatSizeSkipDim(output_shape, output_dim_count - 1);
  const int output_depth = output_shape.Dims(output_dim_count - 1);
  TFLITE_DCHECK_LE(output_depth, filter_shape.Dims(filter_dim_count - 2));
  const int accum_depth = filter_shape.Dims(filter_dim_count - 1);
  for (int b = 0; b < batches; ++b) {
    for (int out_c = 0; out_c < output_depth; ++out_c) {
      AccumScalar acc = 0;
      for (int d = 0; d < accum_depth; ++d) {
        int32_t input_val = input_data[b * accum_depth + d];
        int32_t filter_val = filter_data[out_c * accum_depth + d];
        acc += (filter_val + filter_offset) * input_val;
      }
      if (bias_data) {
        acc += bias_data[out_c];
      }
      int32_t acc_scaled =
          MultiplyByQuantizedMultiplier(acc, output_multiplier, output_shift);
      acc_scaled = std::max(acc_scaled, output_activation_min);
      acc_scaled = std::min(acc_scaled, output_activation_max);
      output_data[out_c + output_depth * b] = static_cast<int16_t>(acc_scaled);
    }
  }
}

}  // namespace reference_integer_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_FULLY_CONNECTED_H_
