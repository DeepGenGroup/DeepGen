/**
 * @file matmul.cpp
 * @author xiebaokang
 * @brief 用于验证新的config下的gemm算法。（带localsplitU）
 * @version 0.1
 * @date 2024-12-18
 * 
 * @copyright Copyright (c) 2024
 * 
 */
#include <iostream>
#include <hip/hip_runtime.h>
#include <algorithm>
#include <vector>
#include <cmath>

template<int VecLen>
__device__ __forceinline__ void VecCpy(float* a, float* b);

template<>
__device__ __forceinline__ void VecCpy<8>(float* a, float* b) {
  (reinterpret_cast<float4*>(a)[0]) = (reinterpret_cast<float4*>(b)[0]);
  (reinterpret_cast<float4*>(a+4)[0]) = (reinterpret_cast<float4*>(b+4)[0]);
}

template<>
__device__ __forceinline__ void VecCpy<6>(float* a, float* b) {
  (reinterpret_cast<float4*>(a)[0]) = (reinterpret_cast<float4*>(b)[0]);
  (reinterpret_cast<float2*>(a+4)[0]) = (reinterpret_cast<float2*>(b+4)[0]);
}

template<>
__device__ __forceinline__ void VecCpy<4>(float* a, float* b) {
  (reinterpret_cast<float4*>(a)[0]) = (reinterpret_cast<float4*>(b)[0]);
}

template<>
__device__ __forceinline__ void VecCpy<2>(float* a, float* b) {
  (reinterpret_cast<float2*>(a)[0]) = (reinterpret_cast<float2*>(b)[0]);
}

template<>
__device__ __forceinline__ void VecCpy<1>(float* a, float* b) {
  (reinterpret_cast<float*>(a)[0]) = (reinterpret_cast<float*>(b)[0]);
} 

template <
  const int BM,
  const int BN,
  const int BK,
  const int TM,
  const int TN,

  const int GLOB_LOAD_WIDTH_A,
  const int GLOB_LOAD_WIDTH_B,

  const int BLOCK_LAYOUT_M,   // BM / TM / WARP_LAYOUT_M
  const int BLOCK_LAYOUT_N,    // BN / TN / WARP_LAYOUT_N
  const int WARP_LAYOUT_M,
  const int WARP_LAYOUT_N,

  const int WARP_SCATTER_WIDTH_A,
  const int WARP_SCATTER_WIDTH_B,
  const int THREAD_SCATTER_WIDTH_A,
  const int THREAD_SCATTER_WIDTH_B,
  
  const int LOCAL_SPLIT_U,   /*2*/
  const int BLOCK_MAPPING,
  const int WARP_SIZE,
  const int GLOB_STORE_WIDTH>
__global__ void  __launch_bounds__(512) matmul(float* A, float* B, float* C, const int M, const int N, const int K) {
  const int BLOCK_Y = BM / TM;
  const int BLOCK_X = BN / TN;
  const int THREAD_NUM = BLOCK_X * BLOCK_Y * LOCAL_SPLIT_U;
  const int tid = threadIdx.x;
  const int tz = tid / (BLOCK_X * BLOCK_Y);
  const int tid_other = tid % (BLOCK_X * BLOCK_Y);

  // enhance L2 cache hit rate
  const int bid = blockIdx.x;
  const int GROUP_NUM = BLOCK_MAPPING * (N / BN);
  const int group_id = bid / GROUP_NUM;
  const int start_y = group_id * BLOCK_MAPPING;
  const int block_mapping = (M / BM) - start_y < BLOCK_MAPPING ? (M / BM) - start_y : BLOCK_MAPPING;
  const int by = start_y + (bid % block_mapping);
  const int bx = (bid % GROUP_NUM) / block_mapping;

  // thread mapping
  const int warp_id = tid_other / WARP_SIZE;
  const int lane_id = tid_other % WARP_SIZE;
  const int warp_y = warp_id / BLOCK_LAYOUT_N;
  const int warp_x = warp_id % BLOCK_LAYOUT_N;
  const int lane_y = lane_id / WARP_LAYOUT_N;
  const int lane_x = lane_id % WARP_LAYOUT_N;

  // split number
  const int BLOCK_REPEAT_A = TM / WARP_SCATTER_WIDTH_A;  // 4 / 2 = 2 
  const int BLOCK_REPEAT_B = TN / WARP_SCATTER_WIDTH_B;  // 6 / 2 = 3
  const int WARP_REPEAT_A = WARP_SCATTER_WIDTH_A / THREAD_SCATTER_WIDTH_A;  // 2 / 2 = 1
  const int WARP_REPEAT_B = WARP_SCATTER_WIDTH_B / THREAD_SCATTER_WIDTH_B;  // 2 / 2 = 1

  // LDSMemory size: BM * BN * 2 > BM * BK * 2 + BN * BK * 2 ? BM * BN * 2 : BM * BK * 2 + BN * BK * 2
  const int LDS_SZIE = (BM * BN * 2) > (BM + BN) * BK * 2 ? (BM * BN * 2) : (BM + BN) * BK * 2;
  __shared__ float LDSMemory[LDS_SZIE];
  float regA[2*TM];
  float regB[2*TN];
  float regC[TM*TN] = {0.0f};

  // global to shread args
  const int GLOB_LOAD_TOTAL_WIDTH_A = BM * BK / THREAD_NUM;  // 8
  const int GLOB_LOAD_TOTAL_WIDTH_B = BN * BK / THREAD_NUM;  // 6

  const int GLOB_LOAD_NUM_A = GLOB_LOAD_TOTAL_WIDTH_A / GLOB_LOAD_WIDTH_A;  // 8 / 6 = 1
  const int GLOB_LOAD_NUM_B = GLOB_LOAD_TOTAL_WIDTH_B / GLOB_LOAD_WIDTH_B;  // 6 / 2 = 3

  const int GLOB_LOAD_REAR_WIDTH_A = GLOB_LOAD_TOTAL_WIDTH_A % GLOB_LOAD_WIDTH_A;  // 8 % 6 = 2
  const int GLOB_LOAD_REAR_WIDTH_B = GLOB_LOAD_TOTAL_WIDTH_B % GLOB_LOAD_WIDTH_B;  // 6 % 2 = 0

  const int GLOB_LOAD_ROW_WIDTH_A = (THREAD_NUM / BK) * GLOB_LOAD_WIDTH_A;  // 48
  const int GLOB_LOAD_ROW_WIDTH_B = (THREAD_NUM / BK) * GLOB_LOAD_WIDTH_B;  // 16

  const int sh_load_row = tid / (THREAD_NUM / BK);
  const int sh_load_col = tid % (THREAD_NUM / BK);

  // mid temp reg
  float tempA[GLOB_LOAD_TOTAL_WIDTH_A];  // 8
  float tempB[GLOB_LOAD_TOTAL_WIDTH_B];  // 6

  // shared to registers arg

  A = &A[by * BM];
  B = &B[bx * BN];

  float* shA = (float *)(LDSMemory);
  float* shB = (float *)(LDSMemory + (BM * BK * 2));

  #pragma unroll
  for (int i=0; i<GLOB_LOAD_NUM_A; i++) {
    // GLOB_LOAD_WIDTH_A = 4 + 2
    VecCpy<GLOB_LOAD_WIDTH_A>(&shA[sh_load_row * BM + 
                                   i * GLOB_LOAD_ROW_WIDTH_A + 
                                   sh_load_col * GLOB_LOAD_WIDTH_A], 
                                &A[sh_load_row * M + 
                                   i * GLOB_LOAD_ROW_WIDTH_A + 
                                   sh_load_col * GLOB_LOAD_WIDTH_A]);
  }
  // rear width = 2
  if (GLOB_LOAD_REAR_WIDTH_A) {
    VecCpy<GLOB_LOAD_REAR_WIDTH_A>(&shA[sh_load_row * BM + 
                                        GLOB_LOAD_NUM_A * GLOB_LOAD_ROW_WIDTH_A + 
                                        sh_load_col * GLOB_LOAD_REAR_WIDTH_A], 
                                     &A[sh_load_row * M + 
                                        GLOB_LOAD_NUM_A * GLOB_LOAD_ROW_WIDTH_A + 
                                        sh_load_col * GLOB_LOAD_REAR_WIDTH_A]);
  }
  
  // globB to sharedB
  #pragma unroll
  for (int i=0; i<GLOB_LOAD_NUM_B; i++) {
    // GLOB_LOAD_WTDTH_B = 2
    VecCpy<GLOB_LOAD_WIDTH_B>(&shB[sh_load_row * BN + 
                                   i * GLOB_LOAD_ROW_WIDTH_B + 
                                   sh_load_col * GLOB_LOAD_WIDTH_B], 
                                &B[sh_load_row * N + 
                                   i * GLOB_LOAD_ROW_WIDTH_B + 
                                   sh_load_col * GLOB_LOAD_WIDTH_B]);
  }
  if (GLOB_LOAD_REAR_WIDTH_B) {
    VecCpy<GLOB_LOAD_REAR_WIDTH_B>(&shB[sh_load_row * BN + 
                                        GLOB_LOAD_NUM_B * GLOB_LOAD_ROW_WIDTH_B + 
                                        sh_load_col * GLOB_LOAD_REAR_WIDTH_B], 
                                     &B[sh_load_row * N + 
                                        GLOB_LOAD_NUM_B * GLOB_LOAD_ROW_WIDTH_B + 
                                        sh_load_col * GLOB_LOAD_REAR_WIDTH_B]);
  }
  __syncthreads();

  #pragma unroll
  for (int i=0; i<BLOCK_REPEAT_A; i++) {
    #pragma unroll
    for (int j=0; j<WARP_REPEAT_A; j++) {
      VecCpy<THREAD_SCATTER_WIDTH_A>(&regA[i * WARP_SCATTER_WIDTH_A + 
                                         j * THREAD_SCATTER_WIDTH_A], 
                                    &shA[tz * BM + 
                                         (i * BLOCK_LAYOUT_M + warp_y) * WARP_LAYOUT_M * WARP_SCATTER_WIDTH_A + 
                                         (j * WARP_LAYOUT_M + lane_y) * THREAD_SCATTER_WIDTH_A]);
    }
  }
  #pragma unroll
  for (int i=0; i<BLOCK_REPEAT_B; i++) {
    #pragma unroll
    for (int j=0; j<WARP_REPEAT_B; j++) {
      VecCpy<THREAD_SCATTER_WIDTH_B>(&regB[i * WARP_SCATTER_WIDTH_B + 
                                         j * THREAD_SCATTER_WIDTH_B], 
                                    &shB[tz * BN + 
                                         (i * BLOCK_LAYOUT_N + warp_x) * WARP_LAYOUT_N * WARP_SCATTER_WIDTH_B + 
                                         (j * WARP_LAYOUT_N + lane_x) * THREAD_SCATTER_WIDTH_B]);
    }
  }

  int write_buffer_id = 1;
  for (int k=BK; k<=K; k+=BK) {

    if (k < K) {   // 最后一轮迭代只有数据计算，没有glob -> shared的操作
      #pragma unroll
      for (int i=0; i<GLOB_LOAD_NUM_A; i++) {
        // GLOB_LOAD_WIDTH_A = 4 + 2
        VecCpy<GLOB_LOAD_WIDTH_A>(&tempA[i * GLOB_LOAD_WIDTH_A], 
                                      &A[(sh_load_row + k) * M + 
                                         i * GLOB_LOAD_ROW_WIDTH_A + 
                                         sh_load_col * GLOB_LOAD_WIDTH_A]);
      }
      // rear width = 2
      if (GLOB_LOAD_REAR_WIDTH_A) {
        VecCpy<GLOB_LOAD_REAR_WIDTH_A>(&tempA[GLOB_LOAD_NUM_A * GLOB_LOAD_WIDTH_A], 
                                           &A[(sh_load_row + k) * M + 
                                              GLOB_LOAD_NUM_A * GLOB_LOAD_ROW_WIDTH_A + 
                                              sh_load_col * GLOB_LOAD_REAR_WIDTH_A]);
      }
      
      // globB to sharedB
      #pragma unroll
      for (int i=0; i<GLOB_LOAD_NUM_B; i++) {
        // GLOB_LOAD_WTDTH_B = 2
        VecCpy<GLOB_LOAD_WIDTH_B>(&tempB[i * GLOB_LOAD_WIDTH_B], 
                                    &B[(sh_load_row + k) * N + 
                                       i * GLOB_LOAD_ROW_WIDTH_B + 
                                       sh_load_col * GLOB_LOAD_WIDTH_B]);
      }
      if (GLOB_LOAD_REAR_WIDTH_B) {
        VecCpy<GLOB_LOAD_REAR_WIDTH_B>(&tempB[GLOB_LOAD_NUM_B * GLOB_LOAD_WIDTH_B], 
                                         &B[(sh_load_row + k) * N + 
                                            GLOB_LOAD_NUM_B * GLOB_LOAD_ROW_WIDTH_B + 
                                            sh_load_col * GLOB_LOAD_REAR_WIDTH_B]);
      }
    }

    int read_buffer_id = write_buffer_id ^ 1;

    #pragma unroll
    for (int bk=0; bk<BK/LOCAL_SPLIT_U-1; bk++) {   // 外积
      #pragma unroll
      for (int i=0; i<BLOCK_REPEAT_A; i++) {
        #pragma unroll
        for (int j=0; j<WARP_REPEAT_A; j++) {
          VecCpy<THREAD_SCATTER_WIDTH_A>(&regA[((bk + 1) % 2) * TM +
                                            i * WARP_SCATTER_WIDTH_A + 
                                            j * THREAD_SCATTER_WIDTH_A], 
                                        &shA[read_buffer_id * BM * BK + 
                                            ((bk + 1) * LOCAL_SPLIT_U + tz) * BM + 
                                            (i * BLOCK_LAYOUT_M + warp_y) * WARP_LAYOUT_M * WARP_SCATTER_WIDTH_A + 
                                            (j * WARP_LAYOUT_M + lane_y) * THREAD_SCATTER_WIDTH_A]);
        }
      }

      #pragma unroll
      for (int i=0; i<BLOCK_REPEAT_B; i++) {
        #pragma unroll
        for (int j=0; j<WARP_REPEAT_B; j++) {
          VecCpy<THREAD_SCATTER_WIDTH_B>(&regB[((bk + 1) % 2) * TN + 
                                             i * WARP_SCATTER_WIDTH_B + 
                                             j * THREAD_SCATTER_WIDTH_B], 
                                        &shB[read_buffer_id * BN * BK + 
                                            ((bk + 1) * LOCAL_SPLIT_U + tz) * BN + 
                                            (i * BLOCK_LAYOUT_N + warp_x) * WARP_LAYOUT_N * WARP_SCATTER_WIDTH_B + 
                                            (j * WARP_LAYOUT_N + lane_x) * THREAD_SCATTER_WIDTH_B]);
        }
      }

      // computing result
      #pragma unroll
      for (int cy=0; cy<TM; cy++) {
        #pragma unroll
        for (int cx=0; cx<TN; cx++) {
          regC[cy * TN + cx] += regA[(bk % 2) * TM + cy] * regB[(bk % 2) * TN + cx];
        }
      }
    }

    if (k < K) {
      #pragma unroll
      for (int i=0; i<GLOB_LOAD_NUM_A; i++) {
        // GLOB_LOAD_WIDTH_A = 4 + 2
        VecCpy<GLOB_LOAD_WIDTH_A>(&shA[write_buffer_id * BM * BK + 
                                       sh_load_row * BM + 
                                       i * GLOB_LOAD_ROW_WIDTH_A + 
                                       sh_load_col * GLOB_LOAD_WIDTH_A], 
                                &tempA[i * GLOB_LOAD_WIDTH_A]);
      }
      // rear width = 2
      if (GLOB_LOAD_REAR_WIDTH_A) {
        VecCpy<GLOB_LOAD_REAR_WIDTH_A>(&shA[write_buffer_id * BM * BK + 
                                            sh_load_row * BM + 
                                            GLOB_LOAD_NUM_A * GLOB_LOAD_ROW_WIDTH_A + 
                                            sh_load_col * GLOB_LOAD_REAR_WIDTH_A], 
                                     &tempA[GLOB_LOAD_NUM_A * GLOB_LOAD_WIDTH_A]);
      }
      
      // globB to sharedB
      #pragma unroll
      for (int i=0; i<GLOB_LOAD_NUM_B; i++) {
        // GLOB_LOAD_WTDTH_B = 2
        VecCpy<GLOB_LOAD_WIDTH_B>(&shB[write_buffer_id * BN * BK + 
                                       sh_load_row * BN + 
                                       i * GLOB_LOAD_ROW_WIDTH_B + 
                                       sh_load_col * GLOB_LOAD_WIDTH_B], 
                                &tempB[i * GLOB_LOAD_WIDTH_B]);
      }
      if (GLOB_LOAD_REAR_WIDTH_B) {
        VecCpy<GLOB_LOAD_REAR_WIDTH_B>(&shB[write_buffer_id * BN * BK + 
                                            sh_load_row * BN + 
                                            GLOB_LOAD_NUM_B * GLOB_LOAD_ROW_WIDTH_B + 
                                            sh_load_col * GLOB_LOAD_REAR_WIDTH_B], 
                                     &tempB[GLOB_LOAD_NUM_B * GLOB_LOAD_WIDTH_B]);
      }
      __syncthreads();
      write_buffer_id ^= 1;
    }

    // last computing result
    #pragma unroll
    for (int cy=0; cy<TM; cy++) {
      #pragma unroll
      for (int cx=0; cx<TN; cx++) {
        regC[cy * TN + cx] += regA[TM + cy] * regB[TN + cx];
      }
    }

    // reg[0] prefatch 
    #pragma unroll
    for (int i=0; i<BLOCK_REPEAT_A; i++) {
      #pragma unroll
      for (int j=0; j<WARP_REPEAT_A; j++) {
        VecCpy<THREAD_SCATTER_WIDTH_A>(&regA[i * WARP_SCATTER_WIDTH_A + 
                                           j * THREAD_SCATTER_WIDTH_A], 
                                      &shA[(read_buffer_id^1) * BM * BK + 
                                           tz * BM + 
                                           (i * BLOCK_LAYOUT_M + warp_y) * WARP_LAYOUT_M * WARP_SCATTER_WIDTH_A + 
                                           (j * WARP_LAYOUT_M + lane_y) * THREAD_SCATTER_WIDTH_A]);
      }
    }
    #pragma unroll
    for (int i=0; i<BLOCK_REPEAT_B; i++) {
      #pragma unroll
      for (int j=0; j<WARP_REPEAT_B; j++) {
        VecCpy<THREAD_SCATTER_WIDTH_B>(&regB[i * WARP_SCATTER_WIDTH_B + 
                                           j * THREAD_SCATTER_WIDTH_B], 
                                      &shB[(read_buffer_id^1) * BN * BK + 
                                           tz * BN + 
                                           (i * BLOCK_LAYOUT_N + warp_x) * WARP_LAYOUT_N * WARP_SCATTER_WIDTH_B + 
                                           (j * WARP_LAYOUT_N + lane_x) * THREAD_SCATTER_WIDTH_B]);
      }
    }
  }

  // regc reduce
  if (LOCAL_SPLIT_U > 1) {
    // reg_c -> shared
    // __syncthreads();
    const int LDS_C_STRIDE = BM * BN;
    const int GLOB_STORE_TOTAL_WIDTH = LDS_C_STRIDE / THREAD_NUM;   // 3072 / 256 = 12
    const int GLOB_STORE_NUM = GLOB_STORE_TOTAL_WIDTH / GLOB_STORE_WIDTH;   // 12 / 2 = 6

    const int GLOB_STORE_COL_THREAD_NUM = BN / GLOB_STORE_WIDTH;
    const int GLOB_STORE_ROW_THREAD_NUM = THREAD_NUM / GLOB_STORE_COL_THREAD_NUM;

    #pragma unroll
    for (int i0=0; i0<BLOCK_REPEAT_A; i0++) {
      #pragma unroll
      for (int i1=0; i1<BLOCK_REPEAT_B; i1++) {
        #pragma unroll
        for (int j0=0; j0<WARP_REPEAT_A; j0++) {
          #pragma unroll
          for (int j1=0; j1<WARP_REPEAT_B; j1++) {
            #pragma unroll 
            for (int k=0; k<THREAD_SCATTER_WIDTH_A; k++) {
              VecCpy<THREAD_SCATTER_WIDTH_B>(&LDSMemory[tz * LDS_C_STRIDE + 
                                                      ((i0 * BLOCK_LAYOUT_M + warp_y) * WARP_LAYOUT_M * WARP_SCATTER_WIDTH_A + (j0 * WARP_LAYOUT_M + lane_y) * THREAD_SCATTER_WIDTH_A + k) * BN + 
                                                      (i1 * BLOCK_LAYOUT_N + warp_x) * WARP_LAYOUT_N * WARP_SCATTER_WIDTH_B + (j1 * WARP_LAYOUT_N + lane_x) * THREAD_SCATTER_WIDTH_B], 
                                                &regC[(i0 * WARP_SCATTER_WIDTH_A + j0 * THREAD_SCATTER_WIDTH_A + k) * TN +
                                                      i1 * WARP_SCATTER_WIDTH_B + j1 * THREAD_SCATTER_WIDTH_B]);
            }
          }
        }
      }
    }
    __syncthreads();

    // shared ->reg
    #pragma unroll
    for (int i=0; i<LOCAL_SPLIT_U; i++) {
      #pragma unroll
      for (int j=0; j<GLOB_STORE_NUM; j++) {
        const int sh_store_col_c = (tid + j * THREAD_NUM) % GLOB_STORE_COL_THREAD_NUM;
        const int sh_store_row_c = (tid + j * THREAD_NUM) / GLOB_STORE_COL_THREAD_NUM;
        if (i == 0) {
          VecCpy<GLOB_STORE_WIDTH>(&regC[j * GLOB_STORE_WIDTH], 
                              &LDSMemory[i * LDS_C_STRIDE + 
                                        sh_store_row_c * BN + 
                                        sh_store_col_c * GLOB_STORE_WIDTH]);
        } else {
          #pragma unroll
          for (int k=0; k<GLOB_STORE_WIDTH; k++) {
            regC[j*GLOB_STORE_WIDTH+k] += LDSMemory[i * LDS_C_STRIDE + 
                                                    sh_store_row_c * BN + 
                                                    sh_store_col_c * GLOB_STORE_WIDTH + k];
          }
        }
      }
    }

    // reg -> global
    #pragma unroll
    for (int i=0; i<GLOB_STORE_NUM; i++) {
      const int sh_store_col_c = (tid + i * THREAD_NUM) % GLOB_STORE_COL_THREAD_NUM;
      const int sh_store_row_c = (tid + i * THREAD_NUM) / GLOB_STORE_COL_THREAD_NUM;
      VecCpy<GLOB_STORE_WIDTH>(&C[(by * BM + sh_store_row_c) * N + 
                                  bx * BN + sh_store_col_c * GLOB_STORE_WIDTH], 
                            &regC[i * GLOB_STORE_WIDTH]);
    }
  } else {
    #pragma unroll
    for (int i0=0; i0<BLOCK_REPEAT_A; i0++) {
      #pragma unroll
      for (int i1=0; i1<BLOCK_REPEAT_B; i1++) {
        #pragma unroll
        for (int j0=0; j0<WARP_REPEAT_A; j0++) {
          #pragma unroll
          for (int j1=0; j1<WARP_REPEAT_B; j1++) {
            #pragma unroll 
            for (int k=0; k<THREAD_SCATTER_WIDTH_A; k++) {
              VecCpy<THREAD_SCATTER_WIDTH_B>(&C[(by * BM + (i0 * BLOCK_LAYOUT_M + warp_y) * WARP_LAYOUT_M * WARP_SCATTER_WIDTH_A + (j0 * WARP_LAYOUT_M + lane_y) * THREAD_SCATTER_WIDTH_A + k) * N + 
                                               bx * BN + (i1 * BLOCK_LAYOUT_N + warp_x) * WARP_LAYOUT_N * WARP_SCATTER_WIDTH_B + (j1 * WARP_LAYOUT_N + lane_x) * THREAD_SCATTER_WIDTH_B], 
                                        &regC[(i0 * WARP_SCATTER_WIDTH_A + j0 * THREAD_SCATTER_WIDTH_A + k) * TN + 
                                               i1 * WARP_SCATTER_WIDTH_B + j1 * THREAD_SCATTER_WIDTH_B]);
            }
          }
        }
      }
    }
  }
}



template <
  const int BM,
  const int BN,
  const int BK,
  const int TM,
  const int TN,

  const int GLOB_LOAD_WIDTH_A,
  const int GLOB_LOAD_WIDTH_B,

  const int BLOCK_LAYOUT_M,   // BM / TM / WARP_LAYOUT_M
  const int BLOCK_LAYOUT_N,    // BN / TN / WARP_LAYOUT_N
  const int WARP_LAYOUT_M,
  const int WARP_LAYOUT_N,

  const int WARP_SCATTER_WIDTH_A,
  const int WARP_SCATTER_WIDTH_B,
  const int THREAD_SCATTER_WIDTH_A,
  const int THREAD_SCATTER_WIDTH_B,
  
  const int LOCAL_SPLIT_U,   /*2*/
  const int BLOCK_MAPPING,
  const int WARP_SIZE,
  const int GLOB_STORE_WIDTH>
__global__ void  __launch_bounds__(512) matmul_1(float* A, float* B, float* C, const int M, const int N, const int K) {
  const int BLOCK_Y = BM / TM;
  const int BLOCK_X = BN / TN;
  const int THREAD_NUM = BLOCK_X * BLOCK_Y * LOCAL_SPLIT_U;
  const int tid = threadIdx.x;
  const int tz = tid / (BLOCK_X * BLOCK_Y);
  const int tid_other = tid % (BLOCK_X * BLOCK_Y);

  // enhance L2 cache hit rate
  const int bid = blockIdx.x;
  const int GROUP_NUM = BLOCK_MAPPING * (N / BN);
  const int group_id = bid / GROUP_NUM;
  const int start_y = group_id * BLOCK_MAPPING;
  const int block_mapping = (M / BM) - start_y < BLOCK_MAPPING ? (M / BM) - start_y : BLOCK_MAPPING;
  const int by = start_y + (bid % block_mapping);
  const int bx = (bid % GROUP_NUM) / block_mapping;

  // thread mapping
  const int warp_id = tid_other / WARP_SIZE;
  const int lane_id = tid_other % WARP_SIZE;
  const int warp_y = warp_id / BLOCK_LAYOUT_N;
  const int warp_x = warp_id % BLOCK_LAYOUT_N;
  const int lane_y = lane_id / WARP_LAYOUT_N;
  const int lane_x = lane_id % WARP_LAYOUT_N;

  // split number
  const int BLOCK_REPEAT_A = TM / WARP_SCATTER_WIDTH_A;  // 4 / 2 = 2 
  const int BLOCK_REPEAT_B = TN / WARP_SCATTER_WIDTH_B;  // 6 / 2 = 3
  const int WARP_REPEAT_A = WARP_SCATTER_WIDTH_A / THREAD_SCATTER_WIDTH_A;  // 2 / 2 = 1
  const int WARP_REPEAT_B = WARP_SCATTER_WIDTH_B / THREAD_SCATTER_WIDTH_B;  // 2 / 2 = 1

  // LDSMemory size: BM * BN * 2 > BM * BK * 2 + BN * BK * 2 ? BM * BN * 2 : BM * BK * 2 + BN * BK * 2
  const int LDS_SZIE = (BM * BN * 2) > (BM + BN) * BK * 2 ? (BM * BN * 2) : (BM + BN) * BK * 2;
  __shared__ float LDSMemory[LDS_SZIE];
  float regA[2*TM];
  float regB[2*TN];
  float regC[TM*TN] = {0.0f};

  // global to shread args
  const int GLOB_LOAD_TOTAL_WIDTH_A = BM * BK / THREAD_NUM;  // 8
  const int GLOB_LOAD_TOTAL_WIDTH_B = BN * BK / THREAD_NUM;  // 6

  const int GLOB_LOAD_NUM_A = GLOB_LOAD_TOTAL_WIDTH_A / GLOB_LOAD_WIDTH_A;  // 8 / 6 = 1
  const int GLOB_LOAD_NUM_B = GLOB_LOAD_TOTAL_WIDTH_B / GLOB_LOAD_WIDTH_B;  // 6 / 2 = 3

  const int GLOB_LOAD_REAR_WIDTH_A = GLOB_LOAD_TOTAL_WIDTH_A % GLOB_LOAD_WIDTH_A;  // 8 % 6 = 2
  const int GLOB_LOAD_REAR_WIDTH_B = GLOB_LOAD_TOTAL_WIDTH_B % GLOB_LOAD_WIDTH_B;  // 6 % 2 = 0

  const int GLOB_LOAD_ROW_WIDTH_A = (THREAD_NUM / BK) * GLOB_LOAD_WIDTH_A;  // 48
  const int GLOB_LOAD_ROW_WIDTH_B = (THREAD_NUM / BK) * GLOB_LOAD_WIDTH_B;  // 16

  const int sh_load_row = tid / (THREAD_NUM / BK);
  const int sh_load_col = tid % (THREAD_NUM / BK);

  // mid temp reg
  float tempA[GLOB_LOAD_TOTAL_WIDTH_A];  // 8
  float tempB[GLOB_LOAD_TOTAL_WIDTH_B];  // 6

  // shared to registers arg

  A = &A[by * BM];
  B = &B[bx * BN];

  float* shA = (float *)(LDSMemory);
  float* shB = (float *)(LDSMemory + (BM * BK * 2));

  #pragma unroll
  for (int i=0; i<GLOB_LOAD_NUM_A; i++) {
    // GLOB_LOAD_WIDTH_A = 4 + 2
    VecCpy<GLOB_LOAD_WIDTH_A>(&shA[sh_load_row * BM + 
                                   i * GLOB_LOAD_ROW_WIDTH_A + 
                                   sh_load_col * GLOB_LOAD_WIDTH_A], 
                                &A[sh_load_row * M + 
                                   i * GLOB_LOAD_ROW_WIDTH_A + 
                                   sh_load_col * GLOB_LOAD_WIDTH_A]);
  }
  // rear width = 2
  if (GLOB_LOAD_REAR_WIDTH_A) {
    VecCpy<GLOB_LOAD_REAR_WIDTH_A>(&shA[sh_load_row * BM + 
                                        GLOB_LOAD_NUM_A * GLOB_LOAD_ROW_WIDTH_A + 
                                        sh_load_col * GLOB_LOAD_REAR_WIDTH_A], 
                                     &A[sh_load_row * M + 
                                        GLOB_LOAD_NUM_A * GLOB_LOAD_ROW_WIDTH_A + 
                                        sh_load_col * GLOB_LOAD_REAR_WIDTH_A]);
  }
  
  // globB to sharedB
  #pragma unroll
  for (int i=0; i<GLOB_LOAD_NUM_B; i++) {
    // GLOB_LOAD_WTDTH_B = 2
    VecCpy<GLOB_LOAD_WIDTH_B>(&shB[sh_load_row * BN + 
                                   i * GLOB_LOAD_ROW_WIDTH_B + 
                                   sh_load_col * GLOB_LOAD_WIDTH_B], 
                                &B[sh_load_row * N + 
                                   i * GLOB_LOAD_ROW_WIDTH_B + 
                                   sh_load_col * GLOB_LOAD_WIDTH_B]);
  }
  if (GLOB_LOAD_REAR_WIDTH_B) {
    VecCpy<GLOB_LOAD_REAR_WIDTH_B>(&shB[sh_load_row * BN + 
                                        GLOB_LOAD_NUM_B * GLOB_LOAD_ROW_WIDTH_B + 
                                        sh_load_col * GLOB_LOAD_REAR_WIDTH_B], 
                                     &B[sh_load_row * N + 
                                        GLOB_LOAD_NUM_B * GLOB_LOAD_ROW_WIDTH_B + 
                                        sh_load_col * GLOB_LOAD_REAR_WIDTH_B]);
  }
  __syncthreads();

  #pragma unroll
  for (int i=0; i<BLOCK_REPEAT_A; i++) {
    #pragma unroll
    for (int j=0; j<WARP_REPEAT_A; j++) {
      VecCpy<THREAD_SCATTER_WIDTH_A>(&regA[i * WARP_SCATTER_WIDTH_A + 
                                         j * THREAD_SCATTER_WIDTH_A], 
                                    &shA[tz * BM + 
                                         (i * BLOCK_LAYOUT_M + warp_y) * WARP_LAYOUT_M * WARP_SCATTER_WIDTH_A + 
                                         (j * WARP_LAYOUT_M + lane_y) * THREAD_SCATTER_WIDTH_A]);
    }
  }
  #pragma unroll
  for (int i=0; i<BLOCK_REPEAT_B; i++) {
    #pragma unroll
    for (int j=0; j<WARP_REPEAT_B; j++) {
      VecCpy<THREAD_SCATTER_WIDTH_B>(&regB[i * WARP_SCATTER_WIDTH_B + 
                                         j * THREAD_SCATTER_WIDTH_B], 
                                    &shB[tz * BN + 
                                         (i * BLOCK_LAYOUT_N + warp_x) * WARP_LAYOUT_N * WARP_SCATTER_WIDTH_B + 
                                         (j * WARP_LAYOUT_N + lane_x) * THREAD_SCATTER_WIDTH_B]);
    }
  }

  int write_buffer_id = 1;
  for (int k=BK; k<=K; k+=BK) {

    if (k < K) {   // 最后一轮迭代只有数据计算，没有glob -> shared的操作
      #pragma unroll
      for (int i=0; i<GLOB_LOAD_NUM_A; i++) {
        // GLOB_LOAD_WIDTH_A = 4 + 2
        VecCpy<GLOB_LOAD_WIDTH_A>(&tempA[i * GLOB_LOAD_WIDTH_A], 
                                      &A[(sh_load_row + k) * M + 
                                         i * GLOB_LOAD_ROW_WIDTH_A + 
                                         sh_load_col * GLOB_LOAD_WIDTH_A]);
      }
      // rear width = 2
      if (GLOB_LOAD_REAR_WIDTH_A) {
        VecCpy<GLOB_LOAD_REAR_WIDTH_A>(&tempA[GLOB_LOAD_NUM_A * GLOB_LOAD_WIDTH_A], 
                                           &A[(sh_load_row + k) * M + 
                                              GLOB_LOAD_NUM_A * GLOB_LOAD_ROW_WIDTH_A + 
                                              sh_load_col * GLOB_LOAD_REAR_WIDTH_A]);
      }
      
      // globB to sharedB
      #pragma unroll
      for (int i=0; i<GLOB_LOAD_NUM_B; i++) {
        // GLOB_LOAD_WTDTH_B = 2
        VecCpy<GLOB_LOAD_WIDTH_B>(&tempB[i * GLOB_LOAD_WIDTH_B], 
                                    &B[(sh_load_row + k) * N + 
                                       i * GLOB_LOAD_ROW_WIDTH_B + 
                                       sh_load_col * GLOB_LOAD_WIDTH_B]);
      }
      if (GLOB_LOAD_REAR_WIDTH_B) {
        VecCpy<GLOB_LOAD_REAR_WIDTH_B>(&tempB[GLOB_LOAD_NUM_B * GLOB_LOAD_WIDTH_B], 
                                         &B[(sh_load_row + k) * N + 
                                            GLOB_LOAD_NUM_B * GLOB_LOAD_ROW_WIDTH_B + 
                                            sh_load_col * GLOB_LOAD_REAR_WIDTH_B]);
      }
    }

    int read_buffer_id = write_buffer_id ^ 1;

    #pragma unroll
    for (int bk=0; bk<BK/LOCAL_SPLIT_U-1; bk++) {   // 外积
      #pragma unroll
      for (int i=0; i<BLOCK_REPEAT_A; i++) {
        #pragma unroll
        for (int j=0; j<WARP_REPEAT_A; j++) {
          VecCpy<THREAD_SCATTER_WIDTH_A>(&regA[((bk + 1) % 2) * TM +
                                            i * WARP_SCATTER_WIDTH_A + 
                                            j * THREAD_SCATTER_WIDTH_A], 
                                        &shA[read_buffer_id * BM * BK + 
                                            ((bk + 1) * LOCAL_SPLIT_U + tz) * BM + 
                                            (i * BLOCK_LAYOUT_M + warp_y) * WARP_LAYOUT_M * WARP_SCATTER_WIDTH_A + 
                                            (j * WARP_LAYOUT_M + lane_y) * THREAD_SCATTER_WIDTH_A]);
        }
      }

      #pragma unroll
      for (int i=0; i<BLOCK_REPEAT_B; i++) {
        #pragma unroll
        for (int j=0; j<WARP_REPEAT_B; j++) {
          VecCpy<THREAD_SCATTER_WIDTH_B>(&regB[((bk + 1) % 2) * TN + 
                                             i * WARP_SCATTER_WIDTH_B + 
                                             j * THREAD_SCATTER_WIDTH_B], 
                                        &shB[read_buffer_id * BN * BK + 
                                            ((bk + 1) * LOCAL_SPLIT_U + tz) * BN + 
                                            (i * BLOCK_LAYOUT_N + warp_x) * WARP_LAYOUT_N * WARP_SCATTER_WIDTH_B + 
                                            (j * WARP_LAYOUT_N + lane_x) * THREAD_SCATTER_WIDTH_B]);
        }
      }

      // computing result
      #pragma unroll
      for (int cy=0; cy<TM; cy++) {
        #pragma unroll
        for (int cx=0; cx<TN; cx++) {
          regC[cy * TN + cx] += regA[(bk % 2) * TM + cy] * regB[(bk % 2) * TN + cx];
        }
      }
    }

    if (k < K) {
      #pragma unroll
      for (int i=0; i<GLOB_LOAD_NUM_A; i++) {
        // GLOB_LOAD_WIDTH_A = 4 + 2
        VecCpy<GLOB_LOAD_WIDTH_A>(&shA[write_buffer_id * BM * BK + 
                                       sh_load_row * BM + 
                                       i * GLOB_LOAD_ROW_WIDTH_A + 
                                       sh_load_col * GLOB_LOAD_WIDTH_A], 
                                &tempA[i * GLOB_LOAD_WIDTH_A]);
      }
      // rear width = 2
      if (GLOB_LOAD_REAR_WIDTH_A) {
        VecCpy<GLOB_LOAD_REAR_WIDTH_A>(&shA[write_buffer_id * BM * BK + 
                                            sh_load_row * BM + 
                                            GLOB_LOAD_NUM_A * GLOB_LOAD_ROW_WIDTH_A + 
                                            sh_load_col * GLOB_LOAD_REAR_WIDTH_A], 
                                     &tempA[GLOB_LOAD_NUM_A * GLOB_LOAD_WIDTH_A]);
      }
      
      // globB to sharedB
      #pragma unroll
      for (int i=0; i<GLOB_LOAD_NUM_B; i++) {
        // GLOB_LOAD_WTDTH_B = 2
        VecCpy<GLOB_LOAD_WIDTH_B>(&shB[write_buffer_id * BN * BK + 
                                       sh_load_row * BN + 
                                       i * GLOB_LOAD_ROW_WIDTH_B + 
                                       sh_load_col * GLOB_LOAD_WIDTH_B], 
                                &tempB[i * GLOB_LOAD_WIDTH_B]);
      }
      if (GLOB_LOAD_REAR_WIDTH_B) {
        VecCpy<GLOB_LOAD_REAR_WIDTH_B>(&shB[write_buffer_id * BN * BK + 
                                            sh_load_row * BN + 
                                            GLOB_LOAD_NUM_B * GLOB_LOAD_ROW_WIDTH_B + 
                                            sh_load_col * GLOB_LOAD_REAR_WIDTH_B], 
                                     &tempB[GLOB_LOAD_NUM_B * GLOB_LOAD_WIDTH_B]);
      }
      __syncthreads();
      write_buffer_id ^= 1;
    }

    // last computing result
    #pragma unroll
    for (int cy=0; cy<TM; cy++) {
      #pragma unroll
      for (int cx=0; cx<TN; cx++) {
        regC[cy * TN + cx] += regA[TM + cy] * regB[TN + cx];
      }
    }

    // reg[0] prefatch 
    #pragma unroll
    for (int i=0; i<BLOCK_REPEAT_A; i++) {
      #pragma unroll
      for (int j=0; j<WARP_REPEAT_A; j++) {
        VecCpy<THREAD_SCATTER_WIDTH_A>(&regA[i * WARP_SCATTER_WIDTH_A + 
                                           j * THREAD_SCATTER_WIDTH_A], 
                                      &shA[(read_buffer_id^1) * BM * BK + 
                                           tz * BM + 
                                           (i * BLOCK_LAYOUT_M + warp_y) * WARP_LAYOUT_M * WARP_SCATTER_WIDTH_A + 
                                           (j * WARP_LAYOUT_M + lane_y) * THREAD_SCATTER_WIDTH_A]);
      }
    }
    #pragma unroll
    for (int i=0; i<BLOCK_REPEAT_B; i++) {
      #pragma unroll
      for (int j=0; j<WARP_REPEAT_B; j++) {
        VecCpy<THREAD_SCATTER_WIDTH_B>(&regB[i * WARP_SCATTER_WIDTH_B + 
                                           j * THREAD_SCATTER_WIDTH_B], 
                                      &shB[(read_buffer_id^1) * BN * BK + 
                                           tz * BN + 
                                           (i * BLOCK_LAYOUT_N + warp_x) * WARP_LAYOUT_N * WARP_SCATTER_WIDTH_B + 
                                           (j * WARP_LAYOUT_N + lane_x) * THREAD_SCATTER_WIDTH_B]);
      }
    }
  }

  // regc reduce
  if (LOCAL_SPLIT_U > 1) {
    // reg_c -> shared
    // __syncthreads();
    const int LDS_C_STRIDE = BM * BN;
    const int GLOB_STORE_TOTAL_WIDTH = LDS_C_STRIDE / THREAD_NUM;   // 3072 / 256 = 12
    const int GLOB_STORE_NUM = GLOB_STORE_TOTAL_WIDTH / GLOB_STORE_WIDTH;   // 12 / 2 = 6
    const int GLOB_STORE_REAR_WIDTH = GLOB_STORE_TOTAL_WIDTH % GLOB_STORE_WIDTH;   // 12 % 6 = 0
    const int GLOB_STORE_ROW_WIDTH = (THREAD_NUM / BM) * GLOB_STORE_WIDTH;   // (256 / 64) * 6 = 24

    const int sh_load_row = tid / (THREAD_NUM / BM);   // [0, 64]
    const int sh_load_col = tid % (THREAD_NUM / BM);   // [0, 4]

    #pragma unroll
    for (int i0=0; i0<BLOCK_REPEAT_A; i0++) {
      #pragma unroll
      for (int i1=0; i1<BLOCK_REPEAT_B; i1++) {
        #pragma unroll
        for (int j0=0; j0<WARP_REPEAT_A; j0++) {
          #pragma unroll
          for (int j1=0; j1<WARP_REPEAT_B; j1++) {
            #pragma unroll 
            for (int k=0; k<THREAD_SCATTER_WIDTH_A; k++) {
              VecCpy<THREAD_SCATTER_WIDTH_B>(&LDSMemory[tz * LDS_C_STRIDE + 
                                                      ((i0 * BLOCK_LAYOUT_M + warp_y) * WARP_LAYOUT_M * WARP_SCATTER_WIDTH_A + (j0 * WARP_LAYOUT_M + lane_y) * THREAD_SCATTER_WIDTH_A + k) * BN + 
                                                      (i1 * BLOCK_LAYOUT_N + warp_x) * WARP_LAYOUT_N * WARP_SCATTER_WIDTH_B + (j1 * WARP_LAYOUT_N + lane_x) * THREAD_SCATTER_WIDTH_B], 
                                                &regC[(i0 * WARP_SCATTER_WIDTH_A + j0 * THREAD_SCATTER_WIDTH_A + k) * TN +
                                                      i1 * WARP_SCATTER_WIDTH_B + j1 * THREAD_SCATTER_WIDTH_B]);
            }
          }
        }
      }
    }
    __syncthreads();

    #pragma unroll
    for (int j=0; j<GLOB_STORE_NUM; j++) {
      VecCpy<GLOB_STORE_WIDTH>(&regC[j * GLOB_STORE_WIDTH], 
                          &LDSMemory[sh_load_row * BN + 
                                    j * GLOB_STORE_ROW_WIDTH + 
                                    sh_load_col * GLOB_STORE_WIDTH]);
    }
    if (GLOB_STORE_REAR_WIDTH) {
      VecCpy<GLOB_STORE_REAR_WIDTH>(&regC[GLOB_STORE_NUM * GLOB_STORE_WIDTH], 
                               &LDSMemory[sh_load_row * BN + 
                                          GLOB_STORE_NUM * GLOB_STORE_ROW_WIDTH + 
                                          sh_load_col * GLOB_STORE_REAR_WIDTH]);
    }
 
    // shared ->reg
    #pragma unroll
    for (int i=1; i<LOCAL_SPLIT_U; i++) {
      #pragma unroll
      for (int j=0; j<GLOB_STORE_NUM; j++) {
        #pragma unroll
        for (int k=0; k<GLOB_STORE_WIDTH; k++) {
          regC[j*GLOB_STORE_WIDTH+k] += LDSMemory[i * LDS_C_STRIDE + 
                                                  sh_load_row * BN + 
                                                  j * GLOB_STORE_ROW_WIDTH + 
                                                  sh_load_col * GLOB_STORE_WIDTH + k];
        }
      }
      if (GLOB_STORE_REAR_WIDTH) {
        #pragma unroll
        for (int k=0; k<GLOB_STORE_WIDTH; k++) {
          regC[GLOB_STORE_NUM * GLOB_STORE_WIDTH + k] += LDSMemory[i * LDS_C_STRIDE + 
                                                                    sh_load_row * BN +
                                                                    GLOB_STORE_NUM * GLOB_STORE_ROW_WIDTH + 
                                                                    sh_load_col * GLOB_STORE_REAR_WIDTH + k];
        }
      }
    }

    // reg -> global
    #pragma unroll
    for (int i=0; i<GLOB_STORE_NUM; i++) {
      VecCpy<GLOB_STORE_WIDTH>(&C[(by * BM + sh_load_row) * N + 
                                  bx * BN + i * GLOB_STORE_ROW_WIDTH + sh_load_col * GLOB_STORE_WIDTH], 
                            &regC[i * GLOB_STORE_WIDTH]);
    }
    if (GLOB_STORE_REAR_WIDTH) {
      VecCpy<GLOB_STORE_REAR_WIDTH>(&C[(by * BM + sh_load_row) * N + 
                                       bx * BN + GLOB_STORE_NUM * GLOB_STORE_ROW_WIDTH + sh_load_col * GLOB_STORE_REAR_WIDTH], 
                                 &regC[GLOB_STORE_NUM * GLOB_STORE_WIDTH]);
    }

  } else {
    #pragma unroll
    for (int i0=0; i0<BLOCK_REPEAT_A; i0++) {
      #pragma unroll
      for (int i1=0; i1<BLOCK_REPEAT_B; i1++) {
        #pragma unroll
        for (int j0=0; j0<WARP_REPEAT_A; j0++) {
          #pragma unroll
          for (int j1=0; j1<WARP_REPEAT_B; j1++) {
            #pragma unroll 
            for (int k=0; k<THREAD_SCATTER_WIDTH_A; k++) {
              VecCpy<THREAD_SCATTER_WIDTH_B>(&C[(by * BM + (i0 * BLOCK_LAYOUT_M + warp_y) * WARP_LAYOUT_M * WARP_SCATTER_WIDTH_A + (j0 * WARP_LAYOUT_M + lane_y) * THREAD_SCATTER_WIDTH_A + k) * N + 
                                               bx * BN + (i1 * BLOCK_LAYOUT_N + warp_x) * WARP_LAYOUT_N * WARP_SCATTER_WIDTH_B + (j1 * WARP_LAYOUT_N + lane_x) * THREAD_SCATTER_WIDTH_B], 
                                        &regC[(i0 * WARP_SCATTER_WIDTH_A + j0 * THREAD_SCATTER_WIDTH_A + k) * TN + 
                                               i1 * WARP_SCATTER_WIDTH_B + j1 * THREAD_SCATTER_WIDTH_B]);
            }
          }
        }
      }
    }
  }
}

template <
  const int BM,
  const int BN,
  const int BK,
  const int TM,
  const int TN,

  const int GLOB_LOAD_WIDTH_A,
  const int GLOB_LOAD_WIDTH_B,

  const int BLOCK_LAYOUT_M,   // BM / TM / WARP_LAYOUT_M
  const int BLOCK_LAYOUT_N,    // BN / TN / WARP_LAYOUT_N
  const int WARP_LAYOUT_M,
  const int WARP_LAYOUT_N,

  const int WARP_SCATTER_WIDTH_A,
  const int WARP_SCATTER_WIDTH_B,
  const int THREAD_SCATTER_WIDTH_A,
  const int THREAD_SCATTER_WIDTH_B,
  
  const int LOCAL_SPLIT_U,   /*2*/
  const int BLOCK_MAPPING,
  const int WARP_SIZE,
  const int GLOB_STORE_WIDTH>
__global__ void  __launch_bounds__(512) matmul_2(float* A, float* B, float* C, const int M, const int N, const int K) {
  // 顺序取 glob to shared
  const int BLOCK_Y = BM / TM;
  const int BLOCK_X = BN / TN;
  const int THREAD_NUM = BLOCK_X * BLOCK_Y * LOCAL_SPLIT_U;
  const int tid = threadIdx.x;
  const int tz = tid / (BLOCK_X * BLOCK_Y);
  const int tid_other = tid % (BLOCK_X * BLOCK_Y);

  // enhance L2 cache hit rate
  const int bid = blockIdx.x;
  const int GROUP_NUM = BLOCK_MAPPING * (N / BN);
  const int group_id = bid / GROUP_NUM;
  const int start_y = group_id * BLOCK_MAPPING;
  const int block_mapping = (M / BM) - start_y < BLOCK_MAPPING ? (M / BM) - start_y : BLOCK_MAPPING;
  const int by = start_y + (bid % block_mapping);
  const int bx = (bid % GROUP_NUM) / block_mapping;

  // thread mapping
  const int warp_id = tid_other / WARP_SIZE;
  const int lane_id = tid_other % WARP_SIZE;
  const int warp_y = warp_id / BLOCK_LAYOUT_N;
  const int warp_x = warp_id % BLOCK_LAYOUT_N;
  const int lane_y = lane_id / WARP_LAYOUT_N;
  const int lane_x = lane_id % WARP_LAYOUT_N;

  // split number
  const int BLOCK_REPEAT_A = TM / WARP_SCATTER_WIDTH_A;  // 4 / 2 = 2 
  const int BLOCK_REPEAT_B = TN / WARP_SCATTER_WIDTH_B;  // 6 / 2 = 3
  const int WARP_REPEAT_A = WARP_SCATTER_WIDTH_A / THREAD_SCATTER_WIDTH_A;  // 2 / 2 = 1
  const int WARP_REPEAT_B = WARP_SCATTER_WIDTH_B / THREAD_SCATTER_WIDTH_B;  // 2 / 2 = 1

  // LDSMemory size: BM * BN * 2 > BM * BK * 2 + BN * BK * 2 ? BM * BN * 2 : BM * BK * 2 + BN * BK * 2
  const int LDS_SZIE = (BM * BN * 2) > (BM + BN) * BK * 2 ? (BM * BN * 2) : (BM + BN) * BK * 2;
  __shared__ float LDSMemory[LDS_SZIE];
  float regA[2*TM];
  float regB[2*TN];
  float regC[TM*TN] = {0.0f};

  // global to shread args
  const int GLOB_LOAD_TOTAL_WIDTH_A = BM * BK / THREAD_NUM;  // 8
  const int GLOB_LOAD_TOTAL_WIDTH_B = BN * BK / THREAD_NUM;  // 6

  const int GLOB_LOAD_NUM_A = GLOB_LOAD_TOTAL_WIDTH_A / GLOB_LOAD_WIDTH_A;  // 8 / 2 = 4
  const int GLOB_LOAD_NUM_B = GLOB_LOAD_TOTAL_WIDTH_B / GLOB_LOAD_WIDTH_B;  // 6 / 2 = 3

  const int GLOB_LOAD_COL_THREAD_NUM_A = BM / GLOB_LOAD_WIDTH_A;  // 64 / 2 = 32
  const int GLOB_LOAD_COL_THREAD_NUM_B = BN / GLOB_LOAD_WIDTH_B;  // 48 / 2 = 24

  const int GLOB_LOAD_ROW_THREAD_NUM_A = THREAD_NUM / GLOB_LOAD_COL_THREAD_NUM_A;
  const int GLOB_LOAD_ROW_THREAD_NUM_B = THREAD_NUM / GLOB_LOAD_COL_THREAD_NUM_B;

  // mid temp reg
  float tempA[GLOB_LOAD_TOTAL_WIDTH_A];  // 8
  float tempB[GLOB_LOAD_TOTAL_WIDTH_B];  // 6

  // shared to registers arg

  A = &A[by * BM];
  B = &B[bx * BN];

  float* shA = (float *)(LDSMemory);
  float* shB = (float *)(LDSMemory + (BM * BK * 2));

  #pragma unroll
  for (int i=0; i<GLOB_LOAD_NUM_A; i++) {
    // GLOB_LOAD_WIDTH_A = 4 + 2
    const int sh_load_row_a = (tid + i * THREAD_NUM) / GLOB_LOAD_COL_THREAD_NUM_A;
    const int sh_load_col_a = (tid + i * THREAD_NUM) % GLOB_LOAD_COL_THREAD_NUM_A;
    VecCpy<GLOB_LOAD_WIDTH_A>(&shA[sh_load_row_a * BM + 
                                   sh_load_col_a * GLOB_LOAD_WIDTH_A], 
                                &A[sh_load_row_a * M + 
                                   sh_load_col_a * GLOB_LOAD_WIDTH_A]);
  }

  
  // globB to sharedB
  #pragma unroll
  for (int i=0; i<GLOB_LOAD_NUM_B; i++) {
    // GLOB_LOAD_WTDTH_B = 2
    const int sh_load_row_b = (tid + i * THREAD_NUM) / GLOB_LOAD_COL_THREAD_NUM_B;
    const int sh_load_col_b = (tid + i * THREAD_NUM) % GLOB_LOAD_COL_THREAD_NUM_B;
    VecCpy<GLOB_LOAD_WIDTH_B>(&shB[sh_load_row_b * BN + 
                                   sh_load_col_b * GLOB_LOAD_WIDTH_B], 
                                &B[sh_load_row_b * N + 
                                   sh_load_col_b * GLOB_LOAD_WIDTH_B]);
  }
  __syncthreads();

  #pragma unroll
  for (int i=0; i<BLOCK_REPEAT_A; i++) {
    #pragma unroll
    for (int j=0; j<WARP_REPEAT_A; j++) {
      VecCpy<THREAD_SCATTER_WIDTH_A>(&regA[i * WARP_SCATTER_WIDTH_A + 
                                         j * THREAD_SCATTER_WIDTH_A], 
                                    &shA[tz * BM + 
                                         (i * BLOCK_LAYOUT_M + warp_y) * WARP_LAYOUT_M * WARP_SCATTER_WIDTH_A + 
                                         (j * WARP_LAYOUT_M + lane_y) * THREAD_SCATTER_WIDTH_A]);
    }
  }
  #pragma unroll
  for (int i=0; i<BLOCK_REPEAT_B; i++) {
    #pragma unroll
    for (int j=0; j<WARP_REPEAT_B; j++) {
      VecCpy<THREAD_SCATTER_WIDTH_B>(&regB[i * WARP_SCATTER_WIDTH_B + 
                                         j * THREAD_SCATTER_WIDTH_B], 
                                    &shB[tz * BN + 
                                         (i * BLOCK_LAYOUT_N + warp_x) * WARP_LAYOUT_N * WARP_SCATTER_WIDTH_B + 
                                         (j * WARP_LAYOUT_N + lane_x) * THREAD_SCATTER_WIDTH_B]);
    }
  }

  int write_buffer_id = 1;
  for (int k=BK; k<=K; k+=BK) {

    if (k < K) {   // 最后一轮迭代只有数据计算，没有glob -> shared的操作
      #pragma unroll
      for (int i=0; i<GLOB_LOAD_NUM_A; i++) {
        // GLOB_LOAD_WIDTH_A = 2
        const int sh_load_row_a = (tid + i * THREAD_NUM) / GLOB_LOAD_COL_THREAD_NUM_A;
        const int sh_load_col_a = (tid + i * THREAD_NUM) % GLOB_LOAD_COL_THREAD_NUM_A;
        VecCpy<GLOB_LOAD_WIDTH_A>(&tempA[i * GLOB_LOAD_WIDTH_A], 
                                      &A[(k + sh_load_row_a) * M + 
                                         sh_load_col_a * GLOB_LOAD_WIDTH_A]);
      }
      
      // globB to sharedB
      #pragma unroll
      for (int i=0; i<GLOB_LOAD_NUM_B; i++) {
        // GLOB_LOAD_WTDTH_B = 2
        const int sh_load_row_b = (tid + i * THREAD_NUM) / GLOB_LOAD_COL_THREAD_NUM_B;
        const int sh_load_col_b = (tid + i * THREAD_NUM) % GLOB_LOAD_COL_THREAD_NUM_B;
        VecCpy<GLOB_LOAD_WIDTH_B>(&tempB[i * GLOB_LOAD_WIDTH_B], 
                                      &B[(k + sh_load_row_b) * N + 
                                         sh_load_col_b * GLOB_LOAD_WIDTH_B]);
      }
    }

    int read_buffer_id = write_buffer_id ^ 1;

    #pragma unroll
    for (int bk=0; bk<BK/LOCAL_SPLIT_U-1; bk++) {   // 外积
      #pragma unroll
      for (int i=0; i<BLOCK_REPEAT_A; i++) {
        #pragma unroll
        for (int j=0; j<WARP_REPEAT_A; j++) {
          VecCpy<THREAD_SCATTER_WIDTH_A>(&regA[((bk + 1) % 2) * TM +
                                            i * WARP_SCATTER_WIDTH_A + 
                                            j * THREAD_SCATTER_WIDTH_A], 
                                        &shA[read_buffer_id * BM * BK + 
                                            ((bk + 1) * LOCAL_SPLIT_U + tz) * BM + 
                                            (i * BLOCK_LAYOUT_M + warp_y) * WARP_LAYOUT_M * WARP_SCATTER_WIDTH_A + 
                                            (j * WARP_LAYOUT_M + lane_y) * THREAD_SCATTER_WIDTH_A]);
        }
      }

      #pragma unroll
      for (int i=0; i<BLOCK_REPEAT_B; i++) {
        #pragma unroll
        for (int j=0; j<WARP_REPEAT_B; j++) {
          VecCpy<THREAD_SCATTER_WIDTH_B>(&regB[((bk + 1) % 2) * TN + 
                                             i * WARP_SCATTER_WIDTH_B + 
                                             j * THREAD_SCATTER_WIDTH_B], 
                                        &shB[read_buffer_id * BN * BK + 
                                            ((bk + 1) * LOCAL_SPLIT_U + tz) * BN + 
                                            (i * BLOCK_LAYOUT_N + warp_x) * WARP_LAYOUT_N * WARP_SCATTER_WIDTH_B + 
                                            (j * WARP_LAYOUT_N + lane_x) * THREAD_SCATTER_WIDTH_B]);
        }
      }

      // computing result
      #pragma unroll
      for (int cy=0; cy<TM; cy++) {
        #pragma unroll
        for (int cx=0; cx<TN; cx++) {
          regC[cy * TN + cx] += regA[(bk % 2) * TM + cy] * regB[(bk % 2) * TN + cx];
        }
      }
    }

    if (k < K) {
      #pragma unroll
      for (int i=0; i<GLOB_LOAD_NUM_A; i++) {
        // GLOB_LOAD_WIDTH_A = 2
        const int sh_load_row_a = (tid + i * THREAD_NUM) / GLOB_LOAD_COL_THREAD_NUM_A;
        const int sh_load_col_a = (tid + i * THREAD_NUM) % GLOB_LOAD_COL_THREAD_NUM_A;
        VecCpy<GLOB_LOAD_WIDTH_A>(&shA[write_buffer_id * BM * BK + 
                                       sh_load_row_a * BM + 
                                       sh_load_col_a * GLOB_LOAD_WIDTH_A], 
                                &tempA[i * GLOB_LOAD_WIDTH_A]);
      }
      
      // globB to sharedB
      #pragma unroll
      for (int i=0; i<GLOB_LOAD_NUM_B; i++) {
        // GLOB_LOAD_WTDTH_B = 2
        const int sh_load_row_b = (tid + i * THREAD_NUM) / GLOB_LOAD_COL_THREAD_NUM_B;
        const int sh_load_col_b = (tid + i * THREAD_NUM) % GLOB_LOAD_COL_THREAD_NUM_B;
        VecCpy<GLOB_LOAD_WIDTH_B>(&shB[write_buffer_id * BN * BK + 
                                       sh_load_row_b * BN + 
                                       sh_load_col_b * GLOB_LOAD_WIDTH_B], 
                                &tempB[i * GLOB_LOAD_WIDTH_B]);
      }
      __syncthreads();
      write_buffer_id ^= 1;
    // }

    // last computing result
    #pragma unroll
    for (int cy=0; cy<TM; cy++) {
      #pragma unroll
      for (int cx=0; cx<TN; cx++) {
        regC[cy * TN + cx] += regA[TM + cy] * regB[TN + cx];
      }
    }

    // reg[0] prefatch 
    #pragma unroll
    for (int i=0; i<BLOCK_REPEAT_A; i++) {
      #pragma unroll
      for (int j=0; j<WARP_REPEAT_A; j++) {
        VecCpy<THREAD_SCATTER_WIDTH_A>(&regA[i * WARP_SCATTER_WIDTH_A + 
                                           j * THREAD_SCATTER_WIDTH_A], 
                                      &shA[(read_buffer_id^1) * BM * BK + 
                                           tz * BM + 
                                           (i * BLOCK_LAYOUT_M + warp_y) * WARP_LAYOUT_M * WARP_SCATTER_WIDTH_A + 
                                           (j * WARP_LAYOUT_M + lane_y) * THREAD_SCATTER_WIDTH_A]);
      }
    }
    #pragma unroll
    for (int i=0; i<BLOCK_REPEAT_B; i++) {
      #pragma unroll
      for (int j=0; j<WARP_REPEAT_B; j++) {
        VecCpy<THREAD_SCATTER_WIDTH_B>(&regB[i * WARP_SCATTER_WIDTH_B + 
                                           j * THREAD_SCATTER_WIDTH_B], 
                                      &shB[(read_buffer_id^1) * BN * BK + 
                                           tz * BN + 
                                           (i * BLOCK_LAYOUT_N + warp_x) * WARP_LAYOUT_N * WARP_SCATTER_WIDTH_B + 
                                           (j * WARP_LAYOUT_N + lane_x) * THREAD_SCATTER_WIDTH_B]);
      }
    }
  }

  // regc reduce
  if (LOCAL_SPLIT_U > 1) {
    // reg_c -> shared
    // __syncthreads();
    const int LDS_C_STRIDE = BM * BN;
    const int GLOB_STORE_TOTAL_WIDTH = LDS_C_STRIDE / THREAD_NUM;   // 3072 / 256 = 12
    const int GLOB_STORE_NUM = GLOB_STORE_TOTAL_WIDTH / GLOB_STORE_WIDTH;   // 12 / 2 = 6

    const int GLOB_STORE_COL_THREAD_NUM = BN / GLOB_STORE_WIDTH;
    const int GLOB_STORE_ROW_THREAD_NUM = THREAD_NUM / GLOB_STORE_COL_THREAD_NUM;

    #pragma unroll
    for (int i0=0; i0<BLOCK_REPEAT_A; i0++) {
      #pragma unroll
      for (int i1=0; i1<BLOCK_REPEAT_B; i1++) {
        #pragma unroll
        for (int j0=0; j0<WARP_REPEAT_A; j0++) {
          #pragma unroll
          for (int j1=0; j1<WARP_REPEAT_B; j1++) {
            #pragma unroll 
            for (int k=0; k<THREAD_SCATTER_WIDTH_A; k++) {
              VecCpy<THREAD_SCATTER_WIDTH_B>(&LDSMemory[tz * LDS_C_STRIDE + 
                                                      ((i0 * BLOCK_LAYOUT_M + warp_y) * WARP_LAYOUT_M * WARP_SCATTER_WIDTH_A + (j0 * WARP_LAYOUT_M + lane_y) * THREAD_SCATTER_WIDTH_A + k) * BN + 
                                                      (i1 * BLOCK_LAYOUT_N + warp_x) * WARP_LAYOUT_N * WARP_SCATTER_WIDTH_B + (j1 * WARP_LAYOUT_N + lane_x) * THREAD_SCATTER_WIDTH_B], 
                                                &regC[(i0 * WARP_SCATTER_WIDTH_A + j0 * THREAD_SCATTER_WIDTH_A + k) * TN +
                                                      i1 * WARP_SCATTER_WIDTH_B + j1 * THREAD_SCATTER_WIDTH_B]);
            }
          }
        }
      }
    }
    __syncthreads();

    // if (tid == 0) {
    //   printf("\n");
    //   for (int i=0; i<2; i++) {
    //     for (int j=0; j<BM; j++) {
    //       // printf("[%d]  ", i*BM+j);
    //       for (int k=0; k<BN; k++) {
    //         printf("%f ", LDSMemory[i*LDS_C_STRIDE + j * BN + k]);
    //       }
    //       printf("\n");
    //     }
    //   }
    // }

    #pragma unroll
    for (int j=0; j<GLOB_STORE_NUM; j++) {
      const int sh_store_col_c = (tid + j * THREAD_NUM) % GLOB_STORE_COL_THREAD_NUM;
      const int sh_store_row_c = (tid + j * THREAD_NUM) / GLOB_STORE_COL_THREAD_NUM;
        VecCpy<GLOB_STORE_WIDTH>(&regC[j * GLOB_STORE_WIDTH], 
                            &LDSMemory[sh_store_row_c * BN + 
                                       sh_store_col_c * GLOB_STORE_WIDTH]);
    }

    // shared ->reg
    #pragma unroll
    for (int i=1; i<LOCAL_SPLIT_U; i++) {
      #pragma unroll
      for (int j=0; j<GLOB_STORE_NUM; j++) {
        const int sh_store_col_c = (tid + j * THREAD_NUM) % GLOB_STORE_COL_THREAD_NUM;
        const int sh_store_row_c = (tid + j * THREAD_NUM) / GLOB_STORE_COL_THREAD_NUM;
        #pragma unroll
        for (int k=0; k<GLOB_STORE_WIDTH; k++) {
          regC[j*GLOB_STORE_WIDTH+k] += LDSMemory[i * LDS_C_STRIDE + 
                                                  sh_store_row_c * BN + 
                                                  sh_store_col_c * GLOB_STORE_WIDTH + k];
        }
      }
    }

    // reg -> global
    #pragma unroll
    for (int i=0; i<GLOB_STORE_NUM; i++) {
      const int sh_store_col_c = (tid + i * THREAD_NUM) % GLOB_STORE_COL_THREAD_NUM;
      const int sh_store_row_c = (tid + i * THREAD_NUM) / GLOB_STORE_COL_THREAD_NUM;
      VecCpy<GLOB_STORE_WIDTH>(&C[(by * BM + sh_store_row_c) * N + 
                                  bx * BN + sh_store_col_c * GLOB_STORE_WIDTH], 
                            &regC[i * GLOB_STORE_WIDTH]);
    }

    // if (tid == 0) {
    //   printf("\n");
    //   for (int j=0; j<BM; j++) {
    //     for (int k=0; k<BN; k++) {
    //       printf("%f ", C[j * BN + k]);
    //     }
    //     printf("\n");
    //   }
    // }


  } else {
    #pragma unroll
    for (int i0=0; i0<BLOCK_REPEAT_A; i0++) {
      #pragma unroll
      for (int i1=0; i1<BLOCK_REPEAT_B; i1++) {
        #pragma unroll
        for (int j0=0; j0<WARP_REPEAT_A; j0++) {
          #pragma unroll
          for (int j1=0; j1<WARP_REPEAT_B; j1++) {
            #pragma unroll 
            for (int k=0; k<THREAD_SCATTER_WIDTH_A; k++) {
              VecCpy<THREAD_SCATTER_WIDTH_B>(&C[(by * BM + (i0 * BLOCK_LAYOUT_M + warp_y) * WARP_LAYOUT_M * WARP_SCATTER_WIDTH_A + (j0 * WARP_LAYOUT_M + lane_y) * THREAD_SCATTER_WIDTH_A + k) * N + 
                                              bx * BN + (i1 * BLOCK_LAYOUT_N + warp_x) * WARP_LAYOUT_N * WARP_SCATTER_WIDTH_B + (j1 * WARP_LAYOUT_N + lane_x) * THREAD_SCATTER_WIDTH_B], 
                                                &regC[(i0 * WARP_SCATTER_WIDTH_A + j0 * THREAD_SCATTER_WIDTH_A + k) * TN + 
                                                      i1 * WARP_SCATTER_WIDTH_B + j1 * THREAD_SCATTER_WIDTH_B]);
            }
          }
        }
      }
    }
  }
}

template <
  const int BM,
  const int BN,
  const int BK,
  const int TM,
  const int TN,

  const int GLOB_LOAD_WIDTH_A,
  const int GLOB_LOAD_WIDTH_B,

  const int BLOCK_LAYOUT_M,   // BM / TM / WARP_LAYOUT_M
  const int BLOCK_LAYOUT_N,    // BN / TN / WARP_LAYOUT_N
  const int WARP_LAYOUT_M,
  const int WARP_LAYOUT_N,

  const int WARP_SCATTER_WIDTH_A,
  const int WARP_SCATTER_WIDTH_B,
  const int THREAD_SCATTER_WIDTH_A,
  const int THREAD_SCATTER_WIDTH_B,
  
  const int LOCAL_SPLIT_U,   /*2*/
  const int BLOCK_MAPPING,
  const int WARP_SIZE,
  const int GLOB_STORE_WIDTH>
__global__ void  __launch_bounds__(512) matmul_3(float* A, float* B, float* C, const int M, const int N, const int K) {
  const int BLOCK_Y = BM / TM;
  const int BLOCK_X = BN / TN;
  const int THREAD_NUM = BLOCK_X * BLOCK_Y * LOCAL_SPLIT_U;
  const int tid = threadIdx.x;
  const int tz = tid / (BLOCK_X * BLOCK_Y);
  const int tid_other = tid % (BLOCK_X * BLOCK_Y);

  // enhance L2 cache hit rate
  const int bid = blockIdx.x;
  const int GROUP_NUM = BLOCK_MAPPING * (N / BN);
  const int group_id = bid / GROUP_NUM;
  const int start_y = group_id * BLOCK_MAPPING;
  const int block_mapping = (M / BM) - start_y < BLOCK_MAPPING ? (M / BM) - start_y : BLOCK_MAPPING;
  const int cross = (bid / block_mapping) % 2;
  const int index = cross == 0 ? (bid % block_mapping) : (-1) * (bid % block_mapping);
  const int by = start_y + cross * (block_mapping - 1) + index;
  const int bx = (bid % GROUP_NUM) / block_mapping;

  // thread mapping
  const int warp_id = tid_other / WARP_SIZE;
  const int lane_id = tid_other % WARP_SIZE;
  const int warp_y = warp_id / BLOCK_LAYOUT_N;
  const int warp_x = warp_id % BLOCK_LAYOUT_N;
  const int lane_y = lane_id / WARP_LAYOUT_N;
  const int lane_x = lane_id % WARP_LAYOUT_N;

  // split number
  const int BLOCK_REPEAT_A = TM / WARP_SCATTER_WIDTH_A;  // 4 / 2 = 2 
  const int BLOCK_REPEAT_B = TN / WARP_SCATTER_WIDTH_B;  // 6 / 2 = 3
  const int WARP_REPEAT_A = WARP_SCATTER_WIDTH_A / THREAD_SCATTER_WIDTH_A;  // 2 / 2 = 1
  const int WARP_REPEAT_B = WARP_SCATTER_WIDTH_B / THREAD_SCATTER_WIDTH_B;  // 2 / 2 = 1

  // LDSMemory size: BM * BN * 2 > BM * BK * 2 + BN * BK * 2 ? BM * BN * 2 : BM * BK * 2 + BN * BK * 2
  const int LDS_SZIE = (BM * BN * 2) > (BM + BN) * BK * 2 ? (BM * BN * 2) : (BM + BN) * BK * 2;
  __shared__ float LDSMemory[LDS_SZIE];
  float regA[2*TM];
  float regB[2*TN];
  float regC[TM*TN] = {0.0f};

  // global to shread args
  const int GLOB_LOAD_TOTAL_WIDTH_A = BM * BK / THREAD_NUM;  // 8
  const int GLOB_LOAD_TOTAL_WIDTH_B = BN * BK / THREAD_NUM;  // 6

  const int GLOB_LOAD_NUM_A = GLOB_LOAD_TOTAL_WIDTH_A / GLOB_LOAD_WIDTH_A;  // 8 / 2 = 4
  const int GLOB_LOAD_NUM_B = GLOB_LOAD_TOTAL_WIDTH_B / GLOB_LOAD_WIDTH_B;  // 6 / 2 = 3

  const int GLOB_LOAD_COL_THREAD_NUM_A = BM / GLOB_LOAD_WIDTH_A;  // 64 / 2 = 32
  const int GLOB_LOAD_COL_THREAD_NUM_B = BN / GLOB_LOAD_WIDTH_B;  // 48 / 2 = 24

  const int GLOB_LOAD_ROW_THREAD_NUM_A = THREAD_NUM / GLOB_LOAD_COL_THREAD_NUM_A;
  const int GLOB_LOAD_ROW_THREAD_NUM_B = THREAD_NUM / GLOB_LOAD_COL_THREAD_NUM_B;

  const int sh_load_row_a = tid / GLOB_LOAD_COL_THREAD_NUM_A;
  const int sh_load_row_b = tid / GLOB_LOAD_COL_THREAD_NUM_B;

  const int sh_load_col_a = tid % GLOB_LOAD_COL_THREAD_NUM_A;
  const int sh_load_col_b = tid % GLOB_LOAD_COL_THREAD_NUM_B;

  // mid temp reg
  float tempA[GLOB_LOAD_TOTAL_WIDTH_A];  // 8
  float tempB[GLOB_LOAD_TOTAL_WIDTH_B];  // 6

  // shared to registers arg

  A = &A[by * BM];
  B = &B[bx * BN];

  float* shA = (float *)(LDSMemory);
  float* shB = (float *)(LDSMemory + (BM * BK * 2));

  #pragma unroll
  for (int i=0; i<GLOB_LOAD_NUM_A; i++) {
    // GLOB_LOAD_WIDTH_A = 4 + 2
    const int sh_load_row_a = (tid + i * THREAD_NUM) / GLOB_LOAD_COL_THREAD_NUM_A;
    const int sh_load_col_a = (tid + i * THREAD_NUM) % GLOB_LOAD_COL_THREAD_NUM_A;
    VecCpy<GLOB_LOAD_WIDTH_A>(&shA[sh_load_row_a * BM + 
                                   sh_load_col_a * GLOB_LOAD_WIDTH_A], 
                                &A[sh_load_row_a * M + 
                                   sh_load_col_a * GLOB_LOAD_WIDTH_A]);
  }

  
  // globB to sharedB
  #pragma unroll
  for (int i=0; i<GLOB_LOAD_NUM_B; i++) {
    // GLOB_LOAD_WTDTH_B = 2
    const int sh_load_row_b = (tid + i * THREAD_NUM) / GLOB_LOAD_COL_THREAD_NUM_B;
    const int sh_load_col_b = (tid + i * THREAD_NUM) % GLOB_LOAD_COL_THREAD_NUM_B;
    VecCpy<GLOB_LOAD_WIDTH_B>(&shB[sh_load_row_b * BN + 
                                   sh_load_col_b * GLOB_LOAD_WIDTH_B], 
                                &B[sh_load_row_b * N + 
                                   sh_load_col_b * GLOB_LOAD_WIDTH_B]);
  }
  __syncthreads();

  #pragma unroll
  for (int i=0; i<BLOCK_REPEAT_A; i++) {
    #pragma unroll
    for (int j=0; j<WARP_REPEAT_A; j++) {
      VecCpy<THREAD_SCATTER_WIDTH_A>(&regA[i * WARP_SCATTER_WIDTH_A + 
                                         j * THREAD_SCATTER_WIDTH_A], 
                                    &shA[tz * BM + 
                                         (i * BLOCK_LAYOUT_M + warp_y) * WARP_LAYOUT_M * WARP_SCATTER_WIDTH_A + 
                                         (j * WARP_LAYOUT_M + lane_y) * THREAD_SCATTER_WIDTH_A]);
    }
  }
  #pragma unroll
  for (int i=0; i<BLOCK_REPEAT_B; i++) {
    #pragma unroll
    for (int j=0; j<WARP_REPEAT_B; j++) {
      VecCpy<THREAD_SCATTER_WIDTH_B>(&regB[i * WARP_SCATTER_WIDTH_B + 
                                         j * THREAD_SCATTER_WIDTH_B], 
                                    &shB[tz * BN + 
                                         (i * BLOCK_LAYOUT_N + warp_x) * WARP_LAYOUT_N * WARP_SCATTER_WIDTH_B + 
                                         (j * WARP_LAYOUT_N + lane_x) * THREAD_SCATTER_WIDTH_B]);
    }
  }

  int write_buffer_id = 1;
  for (int k=BK; k<=K; k+=BK) {

    if (k < K) {   // 最后一轮迭代只有数据计算，没有glob -> shared的操作
      #pragma unroll
      for (int i=0; i<GLOB_LOAD_NUM_A; i++) {
        // GLOB_LOAD_WIDTH_A = 2
        const int sh_load_row_a = (tid + i * THREAD_NUM) / GLOB_LOAD_COL_THREAD_NUM_A;
        const int sh_load_col_a = (tid + i * THREAD_NUM) % GLOB_LOAD_COL_THREAD_NUM_A;
        VecCpy<GLOB_LOAD_WIDTH_A>(&tempA[i * GLOB_LOAD_WIDTH_A], 
                                      &A[(k + sh_load_row_a) * M + 
                                         sh_load_col_a * GLOB_LOAD_WIDTH_A]);
      }
      
      // globB to sharedB
      #pragma unroll
      for (int i=0; i<GLOB_LOAD_NUM_B; i++) {
        // GLOB_LOAD_WTDTH_B = 2
        const int sh_load_row_b = (tid + i * THREAD_NUM) / GLOB_LOAD_COL_THREAD_NUM_B;
        const int sh_load_col_b = (tid + i * THREAD_NUM) % GLOB_LOAD_COL_THREAD_NUM_B;
        VecCpy<GLOB_LOAD_WIDTH_B>(&tempB[i * GLOB_LOAD_WIDTH_B], 
                                      &B[(k + sh_load_row_b) * N + 
                                         sh_load_col_b * GLOB_LOAD_WIDTH_B]);
      }
    }

    int read_buffer_id = write_buffer_id ^ 1;

    #pragma unroll
    for (int bk=0; bk<BK/LOCAL_SPLIT_U-1; bk++) {   // 外积
      #pragma unroll
      for (int i=0; i<BLOCK_REPEAT_A; i++) {
        #pragma unroll
        for (int j=0; j<WARP_REPEAT_A; j++) {
          VecCpy<THREAD_SCATTER_WIDTH_A>(&regA[((bk + 1) % 2) * TM +
                                            i * WARP_SCATTER_WIDTH_A + 
                                            j * THREAD_SCATTER_WIDTH_A], 
                                        &shA[read_buffer_id * BM * BK + 
                                            ((bk + 1) * LOCAL_SPLIT_U + tz) * BM + 
                                            (i * BLOCK_LAYOUT_M + warp_y) * WARP_LAYOUT_M * WARP_SCATTER_WIDTH_A + 
                                            (j * WARP_LAYOUT_M + lane_y) * THREAD_SCATTER_WIDTH_A]);
        }
      }

      #pragma unroll
      for (int i=0; i<BLOCK_REPEAT_B; i++) {
        #pragma unroll
        for (int j=0; j<WARP_REPEAT_B; j++) {
          VecCpy<THREAD_SCATTER_WIDTH_B>(&regB[((bk + 1) % 2) * TN + 
                                             i * WARP_SCATTER_WIDTH_B + 
                                             j * THREAD_SCATTER_WIDTH_B], 
                                        &shB[read_buffer_id * BN * BK + 
                                            ((bk + 1) * LOCAL_SPLIT_U + tz) * BN + 
                                            (i * BLOCK_LAYOUT_N + warp_x) * WARP_LAYOUT_N * WARP_SCATTER_WIDTH_B + 
                                            (j * WARP_LAYOUT_N + lane_x) * THREAD_SCATTER_WIDTH_B]);
        }
      }

      // computing result
      #pragma unroll
      for (int cy=0; cy<TM; cy++) {
        #pragma unroll
        for (int cx=0; cx<TN; cx++) {
          regC[cy * TN + cx] += regA[(bk % 2) * TM + cy] * regB[(bk % 2) * TN + cx];
        }
      }
    }

    if (k < K) {
      #pragma unroll
      for (int i=0; i<GLOB_LOAD_NUM_A; i++) {
        // GLOB_LOAD_WIDTH_A = 2
        const int sh_load_row_a = (tid + i * THREAD_NUM) / GLOB_LOAD_COL_THREAD_NUM_A;
        const int sh_load_col_a = (tid + i * THREAD_NUM) % GLOB_LOAD_COL_THREAD_NUM_A;
        VecCpy<GLOB_LOAD_WIDTH_A>(&shA[write_buffer_id * BM * BK + 
                                       sh_load_row_a * BM + 
                                       sh_load_col_a * GLOB_LOAD_WIDTH_A], 
                                &tempA[i * GLOB_LOAD_WIDTH_A]);
      }
      
      // globB to sharedB
      #pragma unroll
      for (int i=0; i<GLOB_LOAD_NUM_B; i++) {
        // GLOB_LOAD_WTDTH_B = 2
        const int sh_load_row_b = (tid + i * THREAD_NUM) / GLOB_LOAD_COL_THREAD_NUM_B;
        const int sh_load_col_b = (tid + i * THREAD_NUM) % GLOB_LOAD_COL_THREAD_NUM_B;
        VecCpy<GLOB_LOAD_WIDTH_B>(&shB[write_buffer_id * BN * BK + 
                                       sh_load_row_b * BN + 
                                       sh_load_col_b * GLOB_LOAD_WIDTH_B], 
                                &tempB[i * GLOB_LOAD_WIDTH_B]);
      }
      __syncthreads();
      write_buffer_id ^= 1;
    }

    // last computing result
    #pragma unroll
    for (int cy=0; cy<TM; cy++) {
      #pragma unroll
      for (int cx=0; cx<TN; cx++) {
        regC[cy * TN + cx] += regA[TM + cy] * regB[TN + cx];
      }
    }

    // reg[0] prefatch 
    #pragma unroll
    for (int i=0; i<BLOCK_REPEAT_A; i++) {
      #pragma unroll
      for (int j=0; j<WARP_REPEAT_A; j++) {
        VecCpy<THREAD_SCATTER_WIDTH_A>(&regA[i * WARP_SCATTER_WIDTH_A + 
                                           j * THREAD_SCATTER_WIDTH_A], 
                                      &shA[(read_buffer_id^1) * BM * BK + 
                                           tz * BM + 
                                           (i * BLOCK_LAYOUT_M + warp_y) * WARP_LAYOUT_M * WARP_SCATTER_WIDTH_A + 
                                           (j * WARP_LAYOUT_M + lane_y) * THREAD_SCATTER_WIDTH_A]);
      }
    }
    #pragma unroll
    for (int i=0; i<BLOCK_REPEAT_B; i++) {
      #pragma unroll
      for (int j=0; j<WARP_REPEAT_B; j++) {
        VecCpy<THREAD_SCATTER_WIDTH_B>(&regB[i * WARP_SCATTER_WIDTH_B + 
                                           j * THREAD_SCATTER_WIDTH_B], 
                                      &shB[(read_buffer_id^1) * BN * BK + 
                                           tz * BN + 
                                           (i * BLOCK_LAYOUT_N + warp_x) * WARP_LAYOUT_N * WARP_SCATTER_WIDTH_B + 
                                           (j * WARP_LAYOUT_N + lane_x) * THREAD_SCATTER_WIDTH_B]);
      }
    }
  }

  // regc reduce
  if (LOCAL_SPLIT_U > 1) {
    // reg_c -> shared
    const int LDS_C_STRIDE = BM * BN;
    const int GLOB_STORE_TOTAL_WIDTH = LDS_C_STRIDE / THREAD_NUM;   // 3072 / 256 = 12
    const int GLOB_STORE_NUM = GLOB_STORE_TOTAL_WIDTH / GLOB_STORE_WIDTH;   // 12 / 2 = 6

    const int GLOB_STORE_COL_THREAD_NUM = BN / GLOB_STORE_WIDTH;
    const int GLOB_STORE_ROW_THREAD_NUM = THREAD_NUM / GLOB_STORE_COL_THREAD_NUM;

    #pragma unroll
    for (int i0=0; i0<BLOCK_REPEAT_A; i0++) {
      #pragma unroll
      for (int i1=0; i1<BLOCK_REPEAT_B; i1++) {
        #pragma unroll
        for (int j0=0; j0<WARP_REPEAT_A; j0++) {
          #pragma unroll
          for (int j1=0; j1<WARP_REPEAT_B; j1++) {
            #pragma unroll 
            for (int k=0; k<THREAD_SCATTER_WIDTH_A; k++) {
              VecCpy<THREAD_SCATTER_WIDTH_B>(&LDSMemory[tz * LDS_C_STRIDE + 
                                                      ((i0 * BLOCK_LAYOUT_M + warp_y) * WARP_LAYOUT_M * WARP_SCATTER_WIDTH_A + (j0 * WARP_LAYOUT_M + lane_y) * THREAD_SCATTER_WIDTH_A + k) * BN + 
                                                      (i1 * BLOCK_LAYOUT_N + warp_x) * WARP_LAYOUT_N * WARP_SCATTER_WIDTH_B + (j1 * WARP_LAYOUT_N + lane_x) * THREAD_SCATTER_WIDTH_B], 
                                                &regC[(i0 * WARP_SCATTER_WIDTH_A + j0 * THREAD_SCATTER_WIDTH_A + k) * TN +
                                                      i1 * WARP_SCATTER_WIDTH_B + j1 * THREAD_SCATTER_WIDTH_B]);
            }
          }
        }
      }
    }
    __syncthreads();

    // shared ->reg
    #pragma unroll
    for (int i=0; i<LOCAL_SPLIT_U; i++) {
      #pragma unroll
      for (int j=0; j<GLOB_STORE_NUM; j++) {
        const int sh_store_col_c = (tid + j * THREAD_NUM) % GLOB_STORE_COL_THREAD_NUM;
        const int sh_store_row_c = (tid + j * THREAD_NUM) / GLOB_STORE_COL_THREAD_NUM;
        if (i == 0) {
          VecCpy<GLOB_STORE_WIDTH>(&regC[j * GLOB_STORE_WIDTH], 
                              &LDSMemory[i * LDS_C_STRIDE + 
                                        sh_store_row_c * BN + 
                                        sh_store_col_c * GLOB_STORE_WIDTH]);
        } else {
          #pragma unroll
          for (int k=0; k<GLOB_STORE_WIDTH; k++) {
            regC[j*GLOB_STORE_WIDTH+k] += LDSMemory[i * LDS_C_STRIDE + 
                                                    sh_store_row_c * BN + 
                                                    sh_store_col_c * GLOB_STORE_WIDTH + k];
          }
        }
      }
    }

    // reg -> global
    #pragma unroll
    for (int i=0; i<GLOB_STORE_NUM; i++) {
      const int sh_store_col_c = (tid + i * THREAD_NUM) % GLOB_STORE_COL_THREAD_NUM;
      const int sh_store_row_c = (tid + i * THREAD_NUM) / GLOB_STORE_COL_THREAD_NUM;
      VecCpy<GLOB_STORE_WIDTH>(&C[(by * BM + sh_store_row_c) * N + 
                                  bx * BN + sh_store_col_c * GLOB_STORE_WIDTH], 
                            &regC[i * GLOB_STORE_WIDTH]);
    }
  } else {
    #pragma unroll
    for (int i0=0; i0<BLOCK_REPEAT_A; i0++) {
      #pragma unroll
      for (int i1=0; i1<BLOCK_REPEAT_B; i1++) {
        #pragma unroll
        for (int j0=0; j0<WARP_REPEAT_A; j0++) {
          #pragma unroll
          for (int j1=0; j1<WARP_REPEAT_B; j1++) {
            #pragma unroll 
            for (int k=0; k<THREAD_SCATTER_WIDTH_A; k++) {
              VecCpy<THREAD_SCATTER_WIDTH_B>(&C[(by * BM + (i0 * BLOCK_LAYOUT_M + warp_y) * WARP_LAYOUT_M * WARP_SCATTER_WIDTH_A + (j0 * WARP_LAYOUT_M + lane_y) * THREAD_SCATTER_WIDTH_A + k) * N + 
                                              bx * BN + (i1 * BLOCK_LAYOUT_N + warp_x) * WARP_LAYOUT_N * WARP_SCATTER_WIDTH_B + (j1 * WARP_LAYOUT_N + lane_x) * THREAD_SCATTER_WIDTH_B], 
                                                &regC[(i0 * WARP_SCATTER_WIDTH_A + j0 * THREAD_SCATTER_WIDTH_A + k) * TN + 
                                                      i1 * WARP_SCATTER_WIDTH_B + j1 * THREAD_SCATTER_WIDTH_B]);
            }
          }
        }
      }
    }
  }
}


template <
  const int BM,
  const int BN,
  const int BK,
  const int TM,
  const int TN,

  const int GLOB_LOAD_WIDTH_A,
  const int GLOB_LOAD_WIDTH_B,

  const int BLOCK_LAYOUT_M,   // BM / TM / WARP_LAYOUT_M
  const int BLOCK_LAYOUT_N,    // BN / TN / WARP_LAYOUT_N
  const int WARP_LAYOUT_M,
  const int WARP_LAYOUT_N,

  const int WARP_SCATTER_WIDTH_A,
  const int WARP_SCATTER_WIDTH_B,
  const int THREAD_SCATTER_WIDTH_A,
  const int THREAD_SCATTER_WIDTH_B,
  
  const int LOCAL_SPLIT_U,   /*2*/
  const int BLOCK_MAPPING,
  const int WARP_SIZE,
  const int GLOB_STORE_WIDTH>
__global__ void  __launch_bounds__(512) matmul_4(float* A, float* B, float* C, const int M, const int N, const int K) {
  const int BLOCK_Y = BM / TM;
  const int BLOCK_X = BN / TN;
  const int THREAD_NUM = BLOCK_X * BLOCK_Y * LOCAL_SPLIT_U;
  const int tid = threadIdx.x;
  const int tz = tid / (BLOCK_X * BLOCK_Y);
  const int tid_other = tid % (BLOCK_X * BLOCK_Y);

  // enhance L2 cache hit rate
  const int bid = blockIdx.x;
  const int GROUP_NUM = BLOCK_MAPPING * (N / BN);
  const int group_id = bid / GROUP_NUM;
  const int start_y = group_id * BLOCK_MAPPING;
  const int block_mapping = (M / BM) - start_y < BLOCK_MAPPING ? (M / BM) - start_y : BLOCK_MAPPING;
  const int by = start_y + (bid % block_mapping);
  const int bx = (bid % GROUP_NUM) / block_mapping;

  // thread mapping
  const int warp_id = tid_other / WARP_SIZE;
  const int lane_id = tid_other % WARP_SIZE;
  const int warp_y = warp_id / BLOCK_LAYOUT_N;
  const int warp_x = warp_id % BLOCK_LAYOUT_N;
  const int lane_y = lane_id / WARP_LAYOUT_N;
  const int lane_x = lane_id % WARP_LAYOUT_N;

  // split number
  const int BLOCK_REPEAT_A = TM / WARP_SCATTER_WIDTH_A;  // 4 / 2 = 2 
  const int BLOCK_REPEAT_B = TN / WARP_SCATTER_WIDTH_B;  // 6 / 2 = 3
  const int WARP_REPEAT_A = WARP_SCATTER_WIDTH_A / THREAD_SCATTER_WIDTH_A;  // 2 / 2 = 1
  const int WARP_REPEAT_B = WARP_SCATTER_WIDTH_B / THREAD_SCATTER_WIDTH_B;  // 2 / 2 = 1

  // LDSMemory size: BM * BN * 2 > BM * BK * 2 + BN * BK * 2 ? BM * BN * 2 : BM * BK * 2 + BN * BK * 2
  const int LDS_SZIE = (BM * BN * 2) > (BM + BN) * BK * 2 ? (BM * BN * 2) : (BM + BN) * BK * 2;
  __shared__ float LDSMemory[LDS_SZIE];
  float regA[TM];
  float regB[TN];
  float regC[TM*TN] = {0.0f};

  // global to shread args
  const int GLOB_LOAD_TOTAL_WIDTH_A = BM * BK / THREAD_NUM;  // 8
  const int GLOB_LOAD_TOTAL_WIDTH_B = BN * BK / THREAD_NUM;  // 6

  const int GLOB_LOAD_NUM_A = GLOB_LOAD_TOTAL_WIDTH_A / GLOB_LOAD_WIDTH_A;  // 8 / 6 = 1
  const int GLOB_LOAD_NUM_B = GLOB_LOAD_TOTAL_WIDTH_B / GLOB_LOAD_WIDTH_B;  // 6 / 2 = 3

  const int GLOB_LOAD_REAR_WIDTH_A = GLOB_LOAD_TOTAL_WIDTH_A % GLOB_LOAD_WIDTH_A;  // 8 % 6 = 2
  const int GLOB_LOAD_REAR_WIDTH_B = GLOB_LOAD_TOTAL_WIDTH_B % GLOB_LOAD_WIDTH_B;  // 6 % 2 = 0

  const int GLOB_LOAD_ROW_WIDTH_A = (THREAD_NUM / BK) * GLOB_LOAD_WIDTH_A;  // 48
  const int GLOB_LOAD_ROW_WIDTH_B = (THREAD_NUM / BK) * GLOB_LOAD_WIDTH_B;  // 16

  const int sh_load_row = tid / (THREAD_NUM / BK);
  const int sh_load_col = tid % (THREAD_NUM / BK);

  // mid temp reg
  float tempA[GLOB_LOAD_TOTAL_WIDTH_A];  // 8
  float tempB[GLOB_LOAD_TOTAL_WIDTH_B];  // 6

  // shared to registers arg

  A = &A[by * BM];
  B = &B[bx * BN];

  float* shA = (float *)(LDSMemory);
  float* shB = (float *)(LDSMemory + (BM * BK * 2));

  #pragma unroll
  for (int i=0; i<GLOB_LOAD_NUM_A; i++) {
    // GLOB_LOAD_WIDTH_A = 4 + 2
    VecCpy<GLOB_LOAD_WIDTH_A>(&shA[sh_load_row * BM + 
                                   i * GLOB_LOAD_ROW_WIDTH_A + 
                                   sh_load_col * GLOB_LOAD_WIDTH_A], 
                                &A[sh_load_row * M + 
                                   i * GLOB_LOAD_ROW_WIDTH_A + 
                                   sh_load_col * GLOB_LOAD_WIDTH_A]);
  }
  // rear width = 2
  if (GLOB_LOAD_REAR_WIDTH_A) {
    VecCpy<GLOB_LOAD_REAR_WIDTH_A>(&shA[sh_load_row * BM + 
                                        GLOB_LOAD_NUM_A * GLOB_LOAD_ROW_WIDTH_A + 
                                        sh_load_col * GLOB_LOAD_REAR_WIDTH_A], 
                                     &A[sh_load_row * M + 
                                        GLOB_LOAD_NUM_A * GLOB_LOAD_ROW_WIDTH_A + 
                                        sh_load_col * GLOB_LOAD_REAR_WIDTH_A]);
  }
  
  // globB to sharedB
  #pragma unroll
  for (int i=0; i<GLOB_LOAD_NUM_B; i++) {
    // GLOB_LOAD_WTDTH_B = 2
    VecCpy<GLOB_LOAD_WIDTH_B>(&shB[sh_load_row * BN + 
                                   i * GLOB_LOAD_ROW_WIDTH_B + 
                                   sh_load_col * GLOB_LOAD_WIDTH_B], 
                                &B[sh_load_row * N + 
                                   i * GLOB_LOAD_ROW_WIDTH_B + 
                                   sh_load_col * GLOB_LOAD_WIDTH_B]);
  }
  if (GLOB_LOAD_REAR_WIDTH_B) {
    VecCpy<GLOB_LOAD_REAR_WIDTH_B>(&shB[sh_load_row * BN + 
                                        GLOB_LOAD_NUM_B * GLOB_LOAD_ROW_WIDTH_B + 
                                        sh_load_col * GLOB_LOAD_REAR_WIDTH_B], 
                                     &B[sh_load_row * N + 
                                        GLOB_LOAD_NUM_B * GLOB_LOAD_ROW_WIDTH_B + 
                                        sh_load_col * GLOB_LOAD_REAR_WIDTH_B]);
  }
  __syncthreads();

  int write_buffer_id = 1;
  for (int k=BK; k<=K; k+=BK) {

    if (k < K) {   // 最后一轮迭代只有数据计算，没有glob -> shared的操作
      #pragma unroll
      for (int i=0; i<GLOB_LOAD_NUM_A; i++) {
        // GLOB_LOAD_WIDTH_A = 4 + 2
        VecCpy<GLOB_LOAD_WIDTH_A>(&tempA[i * GLOB_LOAD_WIDTH_A], 
                                      &A[(sh_load_row + k) * M + 
                                         i * GLOB_LOAD_ROW_WIDTH_A + 
                                         sh_load_col * GLOB_LOAD_WIDTH_A]);
      }
      // rear width = 2
      if (GLOB_LOAD_REAR_WIDTH_A) {
        VecCpy<GLOB_LOAD_REAR_WIDTH_A>(&tempA[GLOB_LOAD_NUM_A * GLOB_LOAD_WIDTH_A], 
                                           &A[(sh_load_row + k) * M + 
                                              GLOB_LOAD_NUM_A * GLOB_LOAD_ROW_WIDTH_A + 
                                              sh_load_col * GLOB_LOAD_REAR_WIDTH_A]);
      }
      
      // globB to sharedB
      #pragma unroll
      for (int i=0; i<GLOB_LOAD_NUM_B; i++) {
        // GLOB_LOAD_WTDTH_B = 2
        VecCpy<GLOB_LOAD_WIDTH_B>(&tempB[i * GLOB_LOAD_WIDTH_B], 
                                    &B[(sh_load_row + k) * N + 
                                       i * GLOB_LOAD_ROW_WIDTH_B + 
                                       sh_load_col * GLOB_LOAD_WIDTH_B]);
      }
      if (GLOB_LOAD_REAR_WIDTH_B) {
        VecCpy<GLOB_LOAD_REAR_WIDTH_B>(&tempB[GLOB_LOAD_NUM_B * GLOB_LOAD_WIDTH_B], 
                                         &B[(sh_load_row + k) * N + 
                                            GLOB_LOAD_NUM_B * GLOB_LOAD_ROW_WIDTH_B + 
                                            sh_load_col * GLOB_LOAD_REAR_WIDTH_B]);
      }
    }

    int read_buffer_id = write_buffer_id ^ 1;

    #pragma unroll
    for (int bk=0; bk<BK/LOCAL_SPLIT_U; bk++) {   // 外积
      #pragma unroll
      for (int i=0; i<BLOCK_REPEAT_A; i++) {
        #pragma unroll
        for (int j=0; j<WARP_REPEAT_A; j++) {
          VecCpy<THREAD_SCATTER_WIDTH_A>(&regA[i * WARP_SCATTER_WIDTH_A + 
                                            j * THREAD_SCATTER_WIDTH_A], 
                                        &shA[read_buffer_id * BM * BK + 
                                            (bk * LOCAL_SPLIT_U + tz) * BM + 
                                            (i * BLOCK_LAYOUT_M + warp_y) * WARP_LAYOUT_M * WARP_SCATTER_WIDTH_A + 
                                            (j * WARP_LAYOUT_M + lane_y) * THREAD_SCATTER_WIDTH_A]);
        }
      }

      #pragma unroll
      for (int i=0; i<BLOCK_REPEAT_B; i++) {
        #pragma unroll
        for (int j=0; j<WARP_REPEAT_B; j++) {
          VecCpy<THREAD_SCATTER_WIDTH_B>(&regB[i * WARP_SCATTER_WIDTH_B + 
                                             j * THREAD_SCATTER_WIDTH_B], 
                                        &shB[read_buffer_id * BN * BK + 
                                            (bk * LOCAL_SPLIT_U + tz) * BN + 
                                            (i * BLOCK_LAYOUT_N + warp_x) * WARP_LAYOUT_N * WARP_SCATTER_WIDTH_B + 
                                            (j * WARP_LAYOUT_N + lane_x) * THREAD_SCATTER_WIDTH_B]);
        }
      }

      // computing result
      #pragma unroll
      for (int cy=0; cy<TM; cy++) {
        #pragma unroll
        for (int cx=0; cx<TN; cx++) {
          regC[cy * TN + cx] += regA[cy] * regB[cx];
        }
      }
    }

    if (k < K) {
      #pragma unroll
      for (int i=0; i<GLOB_LOAD_NUM_A; i++) {
        // GLOB_LOAD_WIDTH_A = 4 + 2
        VecCpy<GLOB_LOAD_WIDTH_A>(&shA[write_buffer_id * BM * BK + 
                                       sh_load_row * BM + 
                                       i * GLOB_LOAD_ROW_WIDTH_A + 
                                       sh_load_col * GLOB_LOAD_WIDTH_A], 
                                &tempA[i * GLOB_LOAD_WIDTH_A]);
      }
      // rear width = 2
      if (GLOB_LOAD_REAR_WIDTH_A) {
        VecCpy<GLOB_LOAD_REAR_WIDTH_A>(&shA[write_buffer_id * BM * BK + 
                                            sh_load_row * BM + 
                                            GLOB_LOAD_NUM_A * GLOB_LOAD_ROW_WIDTH_A + 
                                            sh_load_col * GLOB_LOAD_REAR_WIDTH_A], 
                                     &tempA[GLOB_LOAD_NUM_A * GLOB_LOAD_WIDTH_A]);
      }
      
      // globB to sharedB
      #pragma unroll
      for (int i=0; i<GLOB_LOAD_NUM_B; i++) {
        // GLOB_LOAD_WTDTH_B = 2
        VecCpy<GLOB_LOAD_WIDTH_B>(&shB[write_buffer_id * BN * BK + 
                                       sh_load_row * BN + 
                                       i * GLOB_LOAD_ROW_WIDTH_B + 
                                       sh_load_col * GLOB_LOAD_WIDTH_B], 
                                &tempB[i * GLOB_LOAD_WIDTH_B]);
      }
      if (GLOB_LOAD_REAR_WIDTH_B) {
        VecCpy<GLOB_LOAD_REAR_WIDTH_B>(&shB[write_buffer_id * BN * BK + 
                                            sh_load_row * BN + 
                                            GLOB_LOAD_NUM_B * GLOB_LOAD_ROW_WIDTH_B + 
                                            sh_load_col * GLOB_LOAD_REAR_WIDTH_B], 
                                     &tempB[GLOB_LOAD_NUM_B * GLOB_LOAD_WIDTH_B]);
      }
      __syncthreads();
      write_buffer_id ^= 1;
    }
  }

  // regc reduce
  if (LOCAL_SPLIT_U > 1) {
    // reg_c -> shared
    const int LDS_C_STRIDE = BM * BN;
    const int GLOB_STORE_TOTAL_WIDTH = LDS_C_STRIDE / THREAD_NUM;   // 3072 / 256 = 12
    const int GLOB_STORE_NUM = GLOB_STORE_TOTAL_WIDTH / GLOB_STORE_WIDTH;   // 12 / 2 = 6
    const int GLOB_STORE_REAR_WIDTH = GLOB_STORE_TOTAL_WIDTH % GLOB_STORE_WIDTH;   // 12 % 6 = 0
    const int GLOB_STORE_ROW_WIDTH = (THREAD_NUM / BM) * GLOB_STORE_WIDTH;   // (256 / 64) * 6 = 24

    const int sh_load_row = tid / (THREAD_NUM / BM);   // [0, 64]
    const int sh_load_col = tid % (THREAD_NUM / BM);   // [0, 4]



    #pragma unroll
    for (int i0=0; i0<BLOCK_REPEAT_A; i0++) {
      #pragma unroll
      for (int i1=0; i1<BLOCK_REPEAT_B; i1++) {
        #pragma unroll
        for (int j0=0; j0<WARP_REPEAT_A; j0++) {
          #pragma unroll
          for (int j1=0; j1<WARP_REPEAT_B; j1++) {
            #pragma unroll 
            for (int k=0; k<THREAD_SCATTER_WIDTH_A; k++) {
              VecCpy<THREAD_SCATTER_WIDTH_B>(&LDSMemory[tz * LDS_C_STRIDE + 
                                                      ((i0 * BLOCK_LAYOUT_M + warp_y) * WARP_LAYOUT_M * WARP_SCATTER_WIDTH_A + (j0 * WARP_LAYOUT_M + lane_y) * THREAD_SCATTER_WIDTH_A + k) * BN + 
                                                      (i1 * BLOCK_LAYOUT_N + warp_x) * WARP_LAYOUT_N * WARP_SCATTER_WIDTH_B + (j1 * WARP_LAYOUT_N + lane_x) * THREAD_SCATTER_WIDTH_B], 
                                                &regC[(i0 * WARP_SCATTER_WIDTH_A + j0 * THREAD_SCATTER_WIDTH_A + k) * TN +
                                                      i1 * WARP_SCATTER_WIDTH_B + j1 * THREAD_SCATTER_WIDTH_B]);
            }
          }
        }
      }
    }
    __syncthreads();

    // shared ->reg
    #pragma unroll
    for (int i=0; i<LOCAL_SPLIT_U; i++) {
      #pragma unroll
      for (int j=0; j<GLOB_STORE_NUM; j++) {
        if (i == 0 ) {
          VecCpy<GLOB_STORE_WIDTH>(&regC[i * GLOB_STORE_TOTAL_WIDTH + 
                                        j * GLOB_STORE_WIDTH], 
                              &LDSMemory[i * LDS_C_STRIDE + 
                                        sh_load_row * BN + 
                                        j * GLOB_STORE_ROW_WIDTH + 
                                        sh_load_col * GLOB_STORE_WIDTH]);
        } else {
          #pragma unroll
          for (int k=0; k<GLOB_STORE_WIDTH; k++) {
            regC[j*GLOB_STORE_WIDTH+k] += LDSMemory[i * LDS_C_STRIDE + 
                                                    sh_load_row * BN + 
                                                    j * GLOB_STORE_ROW_WIDTH + 
                                                    sh_load_col * GLOB_STORE_WIDTH + k];
          }
        }
      }
      if (GLOB_STORE_REAR_WIDTH) {
        if (i == 0) {
          VecCpy<GLOB_STORE_REAR_WIDTH>(&regC[i * GLOB_STORE_TOTAL_WIDTH + 
                                              GLOB_STORE_NUM * GLOB_STORE_WIDTH], 
                                  &LDSMemory[i * LDS_C_STRIDE + 
                                              sh_load_row * BN +
                                              GLOB_STORE_NUM * GLOB_STORE_ROW_WIDTH + 
                                              sh_load_col * GLOB_STORE_REAR_WIDTH]);
        } else {
          #pragma unroll
          for (int k=0; k<GLOB_STORE_WIDTH; k++) {
            regC[GLOB_STORE_NUM * GLOB_STORE_WIDTH + k] += LDSMemory[i * LDS_C_STRIDE + 
                                                                     sh_load_row * BN +
                                                                     GLOB_STORE_NUM * GLOB_STORE_ROW_WIDTH + 
                                                                     sh_load_col * GLOB_STORE_REAR_WIDTH + k];
          }
        }
      }
    }

    // reg -> global
    #pragma unroll
    for (int i=0; i<GLOB_STORE_NUM; i++) {
      VecCpy<GLOB_STORE_WIDTH>(&C[(by * BM + sh_load_row) * N + 
                                  bx * BN + i * GLOB_STORE_ROW_WIDTH + sh_load_col * GLOB_STORE_WIDTH], 
                            &regC[i * GLOB_STORE_WIDTH]);
    }
    if (GLOB_STORE_REAR_WIDTH) {
      VecCpy<GLOB_STORE_REAR_WIDTH>(&C[(by * BM + sh_load_row) * N + 
                                       bx * BN + GLOB_STORE_NUM * GLOB_STORE_ROW_WIDTH + sh_load_col * GLOB_STORE_REAR_WIDTH], 
                                 &regC[GLOB_STORE_NUM * GLOB_STORE_WIDTH]);
    }

  } else {
    #pragma unroll
    for (int i0=0; i0<BLOCK_REPEAT_A; i0++) {
      #pragma unroll
      for (int i1=0; i1<BLOCK_REPEAT_B; i1++) {
        #pragma unroll
        for (int j0=0; j0<WARP_REPEAT_A; j0++) {
          #pragma unroll
          for (int j1=0; j1<WARP_REPEAT_B; j1++) {
            #pragma unroll 
            for (int k=0; k<THREAD_SCATTER_WIDTH_A; k++) {
              VecCpy<THREAD_SCATTER_WIDTH_B>(&C[(by * BM + (i0 * BLOCK_LAYOUT_M + warp_y) * WARP_LAYOUT_M * WARP_SCATTER_WIDTH_A + (j0 * WARP_LAYOUT_M + lane_y) * THREAD_SCATTER_WIDTH_A + k) * N + 
                                               bx * BN + (i1 * BLOCK_LAYOUT_N + warp_x) * WARP_LAYOUT_N * WARP_SCATTER_WIDTH_B + (j1 * WARP_LAYOUT_N + lane_x) * THREAD_SCATTER_WIDTH_B], 
                                        &regC[(i0 * WARP_SCATTER_WIDTH_A + j0 * THREAD_SCATTER_WIDTH_A + k) * TN + 
                                               i1 * WARP_SCATTER_WIDTH_B + j1 * THREAD_SCATTER_WIDTH_B]);
            }
          }
        }
      }
    }
  }
}


template <
  const int BM,
  const int BN,
  const int BK,
  const int TM,
  const int TN,

  const int GLOB_LOAD_WIDTH_A,
  const int GLOB_LOAD_WIDTH_B,

  const int BLOCK_LAYOUT_M,   // BM / TM / WARP_LAYOUT_M
  const int BLOCK_LAYOUT_N,    // BN / TN / WARP_LAYOUT_N
  const int WARP_LAYOUT_M,
  const int WARP_LAYOUT_N,

  const int WARP_SCATTER_WIDTH_A,
  const int WARP_SCATTER_WIDTH_B,
  const int THREAD_SCATTER_WIDTH_A,
  const int THREAD_SCATTER_WIDTH_B,
  
  const int LOCAL_SPLIT_U,   /*2*/
  const int BLOCK_MAPPING,
  const int WARP_SIZE,
  const int GLOB_STORE_WIDTH>
__global__ void  __launch_bounds__(512) matmul_5(float* A, float* B, float* C, const int M, const int N, const int K) {
  const int BLOCK_Y = BM / TM;
  const int BLOCK_X = BN / TN;
  const int THREAD_NUM = BLOCK_X * BLOCK_Y * LOCAL_SPLIT_U;
  const int tid = threadIdx.x;
  const int tz = tid / (BLOCK_X * BLOCK_Y);
  const int tid_other = tid % (BLOCK_X * BLOCK_Y);

  // enhance L2 cache hit rate
  const int bid = blockIdx.x;
  const int GROUP_NUM = BLOCK_MAPPING * (N / BN);
  const int group_id = bid / GROUP_NUM;
  const int start_y = group_id * BLOCK_MAPPING;
  const int block_mapping = (M / BM) - start_y < BLOCK_MAPPING ? (M / BM) - start_y : BLOCK_MAPPING;
  const int by = start_y + (bid % block_mapping);
  const int bx = (bid % GROUP_NUM) / block_mapping;

  // thread mapping
  const int warp_id = tid_other / WARP_SIZE;
  const int lane_id = tid_other % WARP_SIZE;
  const int warp_y = warp_id / BLOCK_LAYOUT_N;
  const int warp_x = warp_id % BLOCK_LAYOUT_N;
  const int lane_y = lane_id / WARP_LAYOUT_N;
  const int lane_x = lane_id % WARP_LAYOUT_N;

  // split number
  const int BLOCK_REPEAT_A = TM / WARP_SCATTER_WIDTH_A;  // 4 / 2 = 2 
  const int BLOCK_REPEAT_B = TN / WARP_SCATTER_WIDTH_B;  // 6 / 2 = 3
  const int WARP_REPEAT_A = WARP_SCATTER_WIDTH_A / THREAD_SCATTER_WIDTH_A;  // 2 / 2 = 1
  const int WARP_REPEAT_B = WARP_SCATTER_WIDTH_B / THREAD_SCATTER_WIDTH_B;  // 2 / 2 = 1

  // LDSMemory size: BM * BN * 2 > BM * BK * 2 + BN * BK * 2 ? BM * BN * 2 : BM * BK * 2 + BN * BK * 2
  const int LDS_SZIE = (BM * BN * 2) > (BM + BN) * BK ? (BM * BN * 2) : (BM + BN) * BK;
  __shared__ float LDSMemory[LDS_SZIE];
  float regA[2*TM];
  float regB[2*TN];
  float regC[TM*TN] = {0.0f};

  // global to shread args
  const int GLOB_LOAD_TOTAL_WIDTH_A = BM * BK / THREAD_NUM;  // 8
  const int GLOB_LOAD_TOTAL_WIDTH_B = BN * BK / THREAD_NUM;  // 6

  const int GLOB_LOAD_NUM_A = GLOB_LOAD_TOTAL_WIDTH_A / GLOB_LOAD_WIDTH_A;  // 8 / 6 = 1
  const int GLOB_LOAD_NUM_B = GLOB_LOAD_TOTAL_WIDTH_B / GLOB_LOAD_WIDTH_B;  // 6 / 2 = 3

  const int GLOB_LOAD_REAR_WIDTH_A = GLOB_LOAD_TOTAL_WIDTH_A % GLOB_LOAD_WIDTH_A;  // 8 % 6 = 2
  const int GLOB_LOAD_REAR_WIDTH_B = GLOB_LOAD_TOTAL_WIDTH_B % GLOB_LOAD_WIDTH_B;  // 6 % 2 = 0

  const int GLOB_LOAD_ROW_WIDTH_A = (THREAD_NUM / BK) * GLOB_LOAD_WIDTH_A;  // 48
  const int GLOB_LOAD_ROW_WIDTH_B = (THREAD_NUM / BK) * GLOB_LOAD_WIDTH_B;  // 16

  const int sh_load_row = tid / (THREAD_NUM / BK);
  const int sh_load_col = tid % (THREAD_NUM / BK);

  // mid temp reg
  float tempA[GLOB_LOAD_TOTAL_WIDTH_A];  // 8
  float tempB[GLOB_LOAD_TOTAL_WIDTH_B];  // 6

  // shared to registers arg

  A = &A[by * BM];
  B = &B[bx * BN];

  float* shA = (float *)(LDSMemory);
  float* shB = (float *)(LDSMemory + (BM * BK));

  for (int k=0; k<K; k+=BK) {
    #pragma unroll
    for (int i=0; i<GLOB_LOAD_NUM_A; i++) {
      __syncthreads();
      // GLOB_LOAD_WIDTH_A = 4 + 2
      VecCpy<GLOB_LOAD_WIDTH_A>(&tempA[i * GLOB_LOAD_WIDTH_A], 
                                    &A[(sh_load_row + k) * M + 
                                        i * GLOB_LOAD_ROW_WIDTH_A + 
                                        sh_load_col * GLOB_LOAD_WIDTH_A]);
    }
    // rear width = 2
    if (GLOB_LOAD_REAR_WIDTH_A) {
      VecCpy<GLOB_LOAD_REAR_WIDTH_A>(&tempA[GLOB_LOAD_NUM_A * GLOB_LOAD_WIDTH_A], 
                                          &A[(sh_load_row + k) * M + 
                                            GLOB_LOAD_NUM_A * GLOB_LOAD_ROW_WIDTH_A + 
                                            sh_load_col * GLOB_LOAD_REAR_WIDTH_A]);
    }
    
    // globB to sharedB
    #pragma unroll
    for (int i=0; i<GLOB_LOAD_NUM_B; i++) {
      // GLOB_LOAD_WTDTH_B = 2
      VecCpy<GLOB_LOAD_WIDTH_B>(&tempB[i * GLOB_LOAD_WIDTH_B], 
                                  &B[(sh_load_row + k) * N + 
                                      i * GLOB_LOAD_ROW_WIDTH_B + 
                                      sh_load_col * GLOB_LOAD_WIDTH_B]);
    }
    if (GLOB_LOAD_REAR_WIDTH_B) {
      VecCpy<GLOB_LOAD_REAR_WIDTH_B>(&tempB[GLOB_LOAD_NUM_B * GLOB_LOAD_WIDTH_B], 
                                        &B[(sh_load_row + k) * N + 
                                          GLOB_LOAD_NUM_B * GLOB_LOAD_ROW_WIDTH_B + 
                                          sh_load_col * GLOB_LOAD_REAR_WIDTH_B]);
    }

    // ------ reg to shared
    #pragma unroll
    for (int i=0; i<GLOB_LOAD_NUM_A; i++) {
      // GLOB_LOAD_WIDTH_A = 4 + 2
      VecCpy<GLOB_LOAD_WIDTH_A>(&shA[sh_load_row * BM + 
                                      i * GLOB_LOAD_ROW_WIDTH_A + 
                                      sh_load_col * GLOB_LOAD_WIDTH_A], 
                              &tempA[i * GLOB_LOAD_WIDTH_A]);
    }
    // rear width = 2
    if (GLOB_LOAD_REAR_WIDTH_A) {
      VecCpy<GLOB_LOAD_REAR_WIDTH_A>(&shA[sh_load_row * BM + 
                                          GLOB_LOAD_NUM_A * GLOB_LOAD_ROW_WIDTH_A + 
                                          sh_load_col * GLOB_LOAD_REAR_WIDTH_A], 
                                    &tempA[GLOB_LOAD_NUM_A * GLOB_LOAD_WIDTH_A]);
    }
    
    // globB to sharedB
    #pragma unroll
    for (int i=0; i<GLOB_LOAD_NUM_B; i++) {
      // GLOB_LOAD_WTDTH_B = 2
      VecCpy<GLOB_LOAD_WIDTH_B>(&shB[sh_load_row * BN + 
                                      i * GLOB_LOAD_ROW_WIDTH_B + 
                                      sh_load_col * GLOB_LOAD_WIDTH_B], 
                              &tempB[i * GLOB_LOAD_WIDTH_B]);
    }
    if (GLOB_LOAD_REAR_WIDTH_B) {
      VecCpy<GLOB_LOAD_REAR_WIDTH_B>(&shB[sh_load_row * BN + 
                                          GLOB_LOAD_NUM_B * GLOB_LOAD_ROW_WIDTH_B + 
                                          sh_load_col * GLOB_LOAD_REAR_WIDTH_B], 
                                    &tempB[GLOB_LOAD_NUM_B * GLOB_LOAD_WIDTH_B]);
    }
    __syncthreads();

    // reg[0] prefatch 
    #pragma unroll
    for (int i=0; i<BLOCK_REPEAT_A; i++) {
      #pragma unroll
      for (int j=0; j<WARP_REPEAT_A; j++) {
        VecCpy<THREAD_SCATTER_WIDTH_A>(&regA[i * WARP_SCATTER_WIDTH_A + 
                                          j * THREAD_SCATTER_WIDTH_A], 
                                      &shA[tz * BM + 
                                          (i * BLOCK_LAYOUT_M + warp_y) * WARP_LAYOUT_M * WARP_SCATTER_WIDTH_A + 
                                          (j * WARP_LAYOUT_M + lane_y) * THREAD_SCATTER_WIDTH_A]);
      }
    }
    #pragma unroll
    for (int i=0; i<BLOCK_REPEAT_B; i++) {
      #pragma unroll
      for (int j=0; j<WARP_REPEAT_B; j++) {
        VecCpy<THREAD_SCATTER_WIDTH_B>(&regB[i * WARP_SCATTER_WIDTH_B + 
                                          j * THREAD_SCATTER_WIDTH_B], 
                                      &shB[tz * BN + 
                                          (i * BLOCK_LAYOUT_N + warp_x) * WARP_LAYOUT_N * WARP_SCATTER_WIDTH_B + 
                                          (j * WARP_LAYOUT_N + lane_x) * THREAD_SCATTER_WIDTH_B]);
      }
    }

    #pragma unroll
    for (int bk=0; bk<BK/LOCAL_SPLIT_U-1; bk++) {   // 外积
      #pragma unroll
      for (int i=0; i<BLOCK_REPEAT_A; i++) {
        #pragma unroll
        for (int j=0; j<WARP_REPEAT_A; j++) {
          VecCpy<THREAD_SCATTER_WIDTH_A>(&regA[((bk + 1) % 2) * TM +
                                            i * WARP_SCATTER_WIDTH_A + 
                                            j * THREAD_SCATTER_WIDTH_A], 
                                        &shA[((bk + 1) * LOCAL_SPLIT_U + tz) * BM + 
                                            (i * BLOCK_LAYOUT_M + warp_y) * WARP_LAYOUT_M * WARP_SCATTER_WIDTH_A + 
                                            (j * WARP_LAYOUT_M + lane_y) * THREAD_SCATTER_WIDTH_A]);
        }
      }

      #pragma unroll
      for (int i=0; i<BLOCK_REPEAT_B; i++) {
        #pragma unroll
        for (int j=0; j<WARP_REPEAT_B; j++) {
          VecCpy<THREAD_SCATTER_WIDTH_B>(&regB[((bk + 1) % 2) * TN + 
                                             i * WARP_SCATTER_WIDTH_B + 
                                             j * THREAD_SCATTER_WIDTH_B], 
                                        &shB[((bk + 1) * LOCAL_SPLIT_U + tz) * BN + 
                                            (i * BLOCK_LAYOUT_N + warp_x) * WARP_LAYOUT_N * WARP_SCATTER_WIDTH_B + 
                                            (j * WARP_LAYOUT_N + lane_x) * THREAD_SCATTER_WIDTH_B]);
        }
      }

      // computing result
      #pragma unroll
      for (int cy=0; cy<TM; cy++) {
        #pragma unroll
        for (int cx=0; cx<TN; cx++) {
          regC[cy * TN + cx] += regA[(bk % 2) * TM + cy] * regB[(bk % 2) * TN + cx];
        }
      }
    }

    // last computing result
    #pragma unroll
    for (int cy=0; cy<TM; cy++) {
      #pragma unroll
      for (int cx=0; cx<TN; cx++) {
        regC[cy * TN + cx] += regA[TM + cy] * regB[TN + cx];
      }
    }
  }

  // regc reduce
  if (LOCAL_SPLIT_U > 1) {
    // reg_c -> shared
    const int LDS_C_STRIDE = BM * BN;
    const int GLOB_STORE_TOTAL_WIDTH = LDS_C_STRIDE / THREAD_NUM;   // 3072 / 256 = 12
    const int GLOB_STORE_NUM = GLOB_STORE_TOTAL_WIDTH / GLOB_STORE_WIDTH;   // 12 / 2 = 6
    const int GLOB_STORE_REAR_WIDTH = GLOB_STORE_TOTAL_WIDTH % GLOB_STORE_WIDTH;   // 12 % 6 = 0
    const int GLOB_STORE_ROW_WIDTH = (THREAD_NUM / BM) * GLOB_STORE_WIDTH;   // (256 / 64) * 6 = 24

    const int sh_load_row = tid / (THREAD_NUM / BM);   // [0, 64]
    const int sh_load_col = tid % (THREAD_NUM / BM);   // [0, 4]



    #pragma unroll
    for (int i0=0; i0<BLOCK_REPEAT_A; i0++) {
      #pragma unroll
      for (int i1=0; i1<BLOCK_REPEAT_B; i1++) {
        #pragma unroll
        for (int j0=0; j0<WARP_REPEAT_A; j0++) {
          #pragma unroll
          for (int j1=0; j1<WARP_REPEAT_B; j1++) {
            #pragma unroll 
            for (int k=0; k<THREAD_SCATTER_WIDTH_A; k++) {
              VecCpy<THREAD_SCATTER_WIDTH_B>(&LDSMemory[tz * LDS_C_STRIDE + 
                                                      ((i0 * BLOCK_LAYOUT_M + warp_y) * WARP_LAYOUT_M * WARP_SCATTER_WIDTH_A + (j0 * WARP_LAYOUT_M + lane_y) * THREAD_SCATTER_WIDTH_A + k) * BN + 
                                                      (i1 * BLOCK_LAYOUT_N + warp_x) * WARP_LAYOUT_N * WARP_SCATTER_WIDTH_B + (j1 * WARP_LAYOUT_N + lane_x) * THREAD_SCATTER_WIDTH_B], 
                                                &regC[(i0 * WARP_SCATTER_WIDTH_A + j0 * THREAD_SCATTER_WIDTH_A + k) * TN +
                                                      i1 * WARP_SCATTER_WIDTH_B + j1 * THREAD_SCATTER_WIDTH_B]);
            }
          }
        }
      }
    }
    __syncthreads();

    // shared ->reg
    #pragma unroll
    for (int i=0; i<LOCAL_SPLIT_U; i++) {
      #pragma unroll
      for (int j=0; j<GLOB_STORE_NUM; j++) {
        if (i == 0 ) {
          VecCpy<GLOB_STORE_WIDTH>(&regC[i * GLOB_STORE_TOTAL_WIDTH + 
                                        j * GLOB_STORE_WIDTH], 
                              &LDSMemory[i * LDS_C_STRIDE + 
                                        sh_load_row * BN + 
                                        j * GLOB_STORE_ROW_WIDTH + 
                                        sh_load_col * GLOB_STORE_WIDTH]);
        } else {
          #pragma unroll
          for (int k=0; k<GLOB_STORE_WIDTH; k++) {
            regC[j*GLOB_STORE_WIDTH+k] += LDSMemory[i * LDS_C_STRIDE + 
                                                    sh_load_row * BN + 
                                                    j * GLOB_STORE_ROW_WIDTH + 
                                                    sh_load_col * GLOB_STORE_WIDTH + k];
          }
        }
      }
      if (GLOB_STORE_REAR_WIDTH) {
        if (i == 0) {
          VecCpy<GLOB_STORE_REAR_WIDTH>(&regC[i * GLOB_STORE_TOTAL_WIDTH + 
                                              GLOB_STORE_NUM * GLOB_STORE_WIDTH], 
                                  &LDSMemory[i * LDS_C_STRIDE + 
                                              sh_load_row * BN +
                                              GLOB_STORE_NUM * GLOB_STORE_ROW_WIDTH + 
                                              sh_load_col * GLOB_STORE_REAR_WIDTH]);
        } else {
          #pragma unroll
          for (int k=0; k<GLOB_STORE_WIDTH; k++) {
            regC[GLOB_STORE_NUM * GLOB_STORE_WIDTH + k] += LDSMemory[i * LDS_C_STRIDE + 
                                                                     sh_load_row * BN +
                                                                     GLOB_STORE_NUM * GLOB_STORE_ROW_WIDTH + 
                                                                     sh_load_col * GLOB_STORE_REAR_WIDTH + k];
          }
        }
      }
    }

    // reg -> global
    #pragma unroll
    for (int i=0; i<GLOB_STORE_NUM; i++) {
      VecCpy<GLOB_STORE_WIDTH>(&C[(by * BM + sh_load_row) * N + 
                                  bx * BN + i * GLOB_STORE_ROW_WIDTH + sh_load_col * GLOB_STORE_WIDTH], 
                            &regC[i * GLOB_STORE_WIDTH]);
    }
    if (GLOB_STORE_REAR_WIDTH) {
      VecCpy<GLOB_STORE_REAR_WIDTH>(&C[(by * BM + sh_load_row) * N + 
                                       bx * BN + GLOB_STORE_NUM * GLOB_STORE_ROW_WIDTH + sh_load_col * GLOB_STORE_REAR_WIDTH], 
                                 &regC[GLOB_STORE_NUM * GLOB_STORE_WIDTH]);
    }

  } else {
    #pragma unroll
    for (int i0=0; i0<BLOCK_REPEAT_A; i0++) {
      #pragma unroll
      for (int i1=0; i1<BLOCK_REPEAT_B; i1++) {
        #pragma unroll
        for (int j0=0; j0<WARP_REPEAT_A; j0++) {
          #pragma unroll
          for (int j1=0; j1<WARP_REPEAT_B; j1++) {
            #pragma unroll 
            for (int k=0; k<THREAD_SCATTER_WIDTH_A; k++) {
              VecCpy<THREAD_SCATTER_WIDTH_B>(&C[(by * BM + (i0 * BLOCK_LAYOUT_M + warp_y) * WARP_LAYOUT_M * WARP_SCATTER_WIDTH_A + (j0 * WARP_LAYOUT_M + lane_y) * THREAD_SCATTER_WIDTH_A + k) * N + 
                                               bx * BN + (i1 * BLOCK_LAYOUT_N + warp_x) * WARP_LAYOUT_N * WARP_SCATTER_WIDTH_B + (j1 * WARP_LAYOUT_N + lane_x) * THREAD_SCATTER_WIDTH_B], 
                                        &regC[(i0 * WARP_SCATTER_WIDTH_A + j0 * THREAD_SCATTER_WIDTH_A + k) * TN + 
                                               i1 * WARP_SCATTER_WIDTH_B + j1 * THREAD_SCATTER_WIDTH_B]);
            }
          }
        }
      }
    }
  }
}


template <typename T>
void display(T *host, int len) {
    // 打印
    int mid = len / 2;
    int start = (rand() % (mid - 1)) + 1;
    int end = (rand() % (mid - 1)) + mid + 1;
    std::cout << "{" << host[0] << ", ..., " << host[start] << ", ..., "  << host[mid] << ", ..., "  << host[end] << ", ..., " << host[len - 1] << "}\n";
}


__global__ void gemm_kernel(float* A, float* B, float* C, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; ++i) {
            sum += A[i*m+row] * B[i*n+col];
        }
        C[row * n + col] = sum;
    }
}

__global__ void verify_kernel(float* C, float* D, int m, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < n) {
      float sub = C[row * n + col] - D[row * n + col];
      if (sub >= 0.00001f || sub <= -0.00001f) {
        printf("error index: (y=%d, x=%d)\nerrer mine: %f   error verify: %f\nsub: %f\n", row, col, C[row * n + col], D[row * n + col], C[row * n + col]-D[row * n + col]);
      }
    }
}

template <typename T>
void hostMatmul(T *A, T *B, T *C, int M, int N, int K) {
    // host矩阵乘
    for (int i=0; i<M; i++) {
      for (int j=0; j<N; j++) {
        for (int k=0; k<K; k++) {
          C[i * N + j] += A[k * M + i] * B[k * N + j];
        }
      }
    }
}

template <typename T>
void verify(T *host, T *device, int M, int N) {
  // 验证
  bool result = true;
  int errorCount = 0;
  for (int i=0; i<M; i++) {
    for (int j=0; j<N; j++) {
      int index = i * N + j;
      if (std::abs(host[index] - device[index]) >= 0.01) {
        printf("error index: (y=%d, x=%d)\n", i, j);
        printf("errer host: %.1f   error device: %.1f\n", host[index], device[index]);
        result = false;
        errorCount++;
      }
    }
  }
  if (result) {
    printf("no error!\n");
  } else {
    printf("have %d error!\n", errorCount);
  }
}


int main() {
  int device_count;
  hipGetDeviceCount(&device_count);
  int device_id;
  hipGetDevice(&device_id);
  hipSetDevice(device_id);

  const int M = 1024;
  const int N = 1056;
  const int K = 1024;

  const int BM = 64;
  const int BN = 48;
  const int BK = 32;
  const int TM = 4;
  const int TN = 6;

  const int GLOB_LOAD_WIDTH_A = 2;
  const int GLOB_LOAD_WIDTH_B = 2;

  const int BLOCK_LAYOUT_M = 2;   // BM / TM / WARP_LAYOUT_M
  const int BLOCK_LAYOUT_N = 1;    // BN / TN / WARP_LAYOUT_N
  const int WARP_LAYOUT_M = 8;
  const int WARP_LAYOUT_N = 8;

  const int WARP_SCATTER_WIDTH_A = 2;
  const int WARP_SCATTER_WIDTH_B = 2;
  const int THREAD_SCATTER_WIDTH_A = 2;
  const int THREAD_SCATTER_WIDTH_B = 2;

  const int LOCAL_SPLIT_U = 2;   /*2*/
  const int BLOCK_MAPPING = 8;
  const int WARP_SIZE = 64;
  const int GLOB_STORE_WIDTH = 2;

  float *A = new float[M * K];
  float *B = new float[N * K];
  float *C = new float[N * M];
  float *D = new float[N * M];
  for (int i = 0; i < M * N; i++) {
    C[i] = 0.0f;
    D[i] = 0.0f;
  }
  for (int i = 0; i < M * K; i++) {
    // if (i % 2) A[i] = 0.0f;
    // else A[i] = 1.0f;
    A[i] = (rand() % 1000) * 0.01f;
  } 
  for (int i = 0; i < N * K; i++) {
    // if (i % 2) B[i] = 0.0f;
    // else B[i] = 1.0f;
    B[i] = (rand() % 1000) * 0.01f;
  }

  float *DA, *DB, *DC, *DD;
  hipMalloc(&DA, M * K * sizeof(float));
  hipMalloc(&DB, N * K * sizeof(float));
  hipMalloc(&DC, N * M * sizeof(float));
  hipMalloc(&DD, N * M * sizeof(float));
  hipMemcpy(DA, A, M * K * sizeof(float), hipMemcpyHostToDevice);
  hipMemcpy(DB, B, N * K * sizeof(float), hipMemcpyHostToDevice);
  
  dim3 grid_size((M/BM)*(N/BN));  // 1024/64=16; 1056/48=22
  dim3 block_size(((BM/TM)*(BN/TN)) * LOCAL_SPLIT_U);  // 64/4=16; 48/6=8

  dim3 dimBlock = {16, 16};
  dim3 dimGrid = {(N + dimBlock.x - 1) / dimBlock.x, (M + dimBlock.y - 1) / dimBlock.y};


  std::vector<float> costs;
  for (int i=0; i<1; i++) {
    // 执行内核函数
    hipEvent_t startEvent, stopEvent;
    hipEventCreate(&startEvent);
    hipEventCreate(&stopEvent);
    hipEventRecord(startEvent, 0);

    // 前面的加载使用非连续，后面的reduceC使用连续
    // matmul<BM, BN, BK, TM, TN, 
    //   GLOB_LOAD_WIDTH_A, GLOB_LOAD_WIDTH_B,
    //   BLOCK_LAYOUT_M, BLOCK_LAYOUT_N,
    //   WARP_LAYOUT_M, WARP_LAYOUT_N, 
    //   WARP_SCATTER_WIDTH_A, WARP_SCATTER_WIDTH_B, 
    //   THREAD_SCATTER_WIDTH_A, THREAD_SCATTER_WIDTH_B,
    //   LOCAL_SPLIT_U, BLOCK_MAPPING, WARP_SIZE, GLOB_STORE_WIDTH><<<grid_size, block_size>>>(DA, DB, DC, M, N, K);

    // 在origin的基础上修改，修改splitu存储方式，直接从shared加到reg上
    matmul_1<BM, BN, BK, TM, TN, 
      GLOB_LOAD_WIDTH_A, GLOB_LOAD_WIDTH_B,
      BLOCK_LAYOUT_M, BLOCK_LAYOUT_N,
      WARP_LAYOUT_M, WARP_LAYOUT_N, 
      WARP_SCATTER_WIDTH_A, WARP_SCATTER_WIDTH_B, 
      THREAD_SCATTER_WIDTH_A, THREAD_SCATTER_WIDTH_B,
      LOCAL_SPLIT_U, BLOCK_MAPPING, WARP_SIZE, GLOB_STORE_WIDTH><<<grid_size, block_size>>>(DA, DB, DC, M, N, K);
    
    // 在matmul_1的基础上修改，讲所有glob load的方式改为顺序load
    // 这个有限制，GLOB_LOAD_WIDTH_A/B必须为BM/BN和GLOB_LOAD_TATOL_WIDTH_A/B的约数
    // GLOB_STORE_WIDTH必须为BN和GLOB_LOAD_TATOL_WIDTH_C的约数
    // matmul_2<BM, BN, BK, TM, TN, 
      // GLOB_LOAD_WIDTH_A, GLOB_LOAD_WIDTH_B,
      // BLOCK_LAYOUT_M, BLOCK_LAYOUT_N,
      // WARP_LAYOUT_M, WARP_LAYOUT_N, 
      // WARP_SCATTER_WIDTH_A, WARP_SCATTER_WIDTH_B, 
      // THREAD_SCATTER_WIDTH_A, THREAD_SCATTER_WIDTH_B,
      // LOCAL_SPLIT_U, BLOCK_MAPPING, WARP_SIZE, GLOB_STORE_WIDTH><<<grid_size, block_size>>>(DA, DB, DC, M, N, K);

    // 在matmul_2的基础上修改，将L2 cache的方式修改为s形
    // matmul_3<BM, BN, BK, TM, TN, 
    //   GLOB_LOAD_WIDTH_A, GLOB_LOAD_WIDTH_B,
    //   BLOCK_LAYOUT_M, BLOCK_LAYOUT_N,
    //   WARP_LAYOUT_M, WARP_LAYOUT_N, 
    //   WARP_SCATTER_WIDTH_A, WARP_SCATTER_WIDTH_B, 
    //   THREAD_SCATTER_WIDTH_A, THREAD_SCATTER_WIDTH_B,
    //   LOCAL_SPLIT_U, BLOCK_MAPPING, WARP_SIZE, GLOB_STORE_WIDTH><<<grid_size, block_size>>>(DA, DB, DC, M, N, K);
  
    // 在matmul_1的基础上修改，只做shared的流水
    // matmul_4<BM, BN, BK, TM, TN, 
    //   GLOB_LOAD_WIDTH_A, GLOB_LOAD_WIDTH_B,
    //   BLOCK_LAYOUT_M, BLOCK_LAYOUT_N,
    //   WARP_LAYOUT_M, WARP_LAYOUT_N, 
    //   WARP_SCATTER_WIDTH_A, WARP_SCATTER_WIDTH_B, 
    //   THREAD_SCATTER_WIDTH_A, THREAD_SCATTER_WIDTH_B,
    //   LOCAL_SPLIT_U, BLOCK_MAPPING, WARP_SIZE, GLOB_STORE_WIDTH><<<grid_size, block_size>>>(DA, DB, DC, M, N, K);
    

    // 在matmul_1的基础上修改，只做reg的流水
    // matmul_5<BM, BN, BK, TM, TN, 
    //   GLOB_LOAD_WIDTH_A, GLOB_LOAD_WIDTH_B,
    //   BLOCK_LAYOUT_M, BLOCK_LAYOUT_N,
    //   WARP_LAYOUT_M, WARP_LAYOUT_N, 
    //   WARP_SCATTER_WIDTH_A, WARP_SCATTER_WIDTH_B, 
    //   THREAD_SCATTER_WIDTH_A, THREAD_SCATTER_WIDTH_B,
    //   LOCAL_SPLIT_U, BLOCK_MAPPING, WARP_SIZE, GLOB_STORE_WIDTH><<<grid_size, block_size>>>(DA, DB, DC, M, N, K);

    // gemm_kernel<<<dimGrid, dimBlock>>>(DA, DB, DC, M, N, K);
    gemm_kernel<<<dimGrid, dimBlock>>>(DA, DB, DD, M, N, K);
    verify_kernel<<<dimGrid, dimBlock>>>(DC, DD, M, N);

    hipEventRecord(stopEvent, 0);
    hipEventSynchronize(stopEvent);

    float elapsedTime;
    hipEventElapsedTime(&elapsedTime, startEvent, stopEvent);
    costs.push_back(elapsedTime);
  }

  std::sort(costs.begin(), costs.end());
  float time = costs[costs.size()/2];
  double tflops = (2 * static_cast<uint64_t>(M) * N * K) / (time / 1000) / 1e12;

  // hipMemcpy(C, DC, M * N * sizeof(float), hipMemcpyDeviceToHost);
  // hipMemcpy(D, DD, M * N * sizeof(float), hipMemcpyDeviceToHost);
  std::cout << "time cost: " << time << "ms\n";
  std::cout << "tflops: " << tflops << std::endl;
  // display(C, M * N);
  // hostMatmul(A, B, D, M, N, K);
  // verify(D, C, M, N);

  hipFree(DA);
  hipFree(DB);
  hipFree(DC);
  hipFree(DD);
  delete[] A;
  delete[] B;
  delete[] C;
  delete[] D;
  return 0;
}