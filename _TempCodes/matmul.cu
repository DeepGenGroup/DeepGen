#include <cuda_runtime.h>
#include <iostream>
#include <algorithm>
#include <vector>
#include <cmath>
#include <cublas_v2.h>

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
  const int GLOB_STORE_WIDTH
>
__global__ void  __launch_bounds__(512) matmul_adjust(float* A, float* B, float* C, const int M, const int N, const int K) {
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

  // prefetch load glob to shared
  #pragma unroll
  for (int i=0; i<GLOB_LOAD_NUM_A; i++) {
    VecCpy<GLOB_LOAD_WIDTH_A>(&shA[sh_load_row * BM + 
                                   i * GLOB_LOAD_ROW_WIDTH_A + 
                                   sh_load_col * GLOB_LOAD_WIDTH_A], 
                                &A[sh_load_row * M + 
                                   i * GLOB_LOAD_ROW_WIDTH_A + 
                                   sh_load_col * GLOB_LOAD_WIDTH_A]);
  }
  #pragma unroll
  for (int i=0; i<GLOB_LOAD_NUM_B; i++) {
    VecCpy<GLOB_LOAD_WIDTH_B>(&shB[sh_load_row * BN + 
                                   i * GLOB_LOAD_ROW_WIDTH_B + 
                                   sh_load_col * GLOB_LOAD_WIDTH_B], 
                                &B[sh_load_row * N + 
                                   i * GLOB_LOAD_ROW_WIDTH_B + 
                                   sh_load_col * GLOB_LOAD_WIDTH_B]);
  }
  __syncthreads();
  // prefetch load shared to reg
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

  // for k
  int write_buffer_id = 1;
  for (int k=BK; k<=K; k+=BK) {

    if (k < K) {  // 最后一轮迭代只有数据计算，没有glob -> shared的操作
      #pragma unroll
      for (int i=0; i<GLOB_LOAD_NUM_A; i++) {
        VecCpy<GLOB_LOAD_WIDTH_A>(&tempA[i * GLOB_LOAD_WIDTH_A], 
                                      &A[(sh_load_row + k) * M + 
                                         i * GLOB_LOAD_ROW_WIDTH_A + 
                                         sh_load_col * GLOB_LOAD_WIDTH_A]);
      }
      #pragma unroll
      for (int i=0; i<GLOB_LOAD_NUM_B; i++) {
        VecCpy<GLOB_LOAD_WIDTH_B>(&tempB[i * GLOB_LOAD_WIDTH_B], 
                                    &B[(sh_load_row + k) * N + 
                                       i * GLOB_LOAD_ROW_WIDTH_B + 
                                       sh_load_col * GLOB_LOAD_WIDTH_B]);
      }
    }
    // for bk
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

      // calculate result
      #pragma unroll
      for (int cy=0; cy<TM; cy++) {
        #pragma unroll
        for (int cx=0; cx<TN; cx++) {
          regC[cy * TN + cx] += regA[(bk % 2) * TM + cy] * regB[(bk % 2) * TN + cx];
        }
      }
    }

    if (k < K) {  // 最后只有数据计算
      #pragma unroll
      for (int i=0; i<GLOB_LOAD_NUM_A; i++) {
        VecCpy<GLOB_LOAD_WIDTH_A>(&shA[write_buffer_id * BM * BK + 
                                       sh_load_row * BM + 
                                       i * GLOB_LOAD_ROW_WIDTH_A + 
                                       sh_load_col * GLOB_LOAD_WIDTH_A], 
                                &tempA[i * GLOB_LOAD_WIDTH_A]);
      }
      #pragma unroll
      for (int i=0; i<GLOB_LOAD_NUM_B; i++) {
        VecCpy<GLOB_LOAD_WIDTH_B>(&shB[write_buffer_id * BN * BK + 
                                       sh_load_row * BN + 
                                       i * GLOB_LOAD_ROW_WIDTH_B + 
                                       sh_load_col * GLOB_LOAD_WIDTH_B], 
                                &tempB[i * GLOB_LOAD_WIDTH_B]);
      }
      __syncthreads();
      write_buffer_id ^= 1;
    }

    // last calculate result
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

  if (LOCAL_SPLIT_U > 1) {
    __syncthreads();
    const int LDS_C_STRIDE = BM * BN;
    const int GLOB_STORE_TOTAL_WIDTH = LDS_C_STRIDE / THREAD_NUM;   // 3072 / 256 = 12
    const int GLOB_STORE_NUM = GLOB_STORE_TOTAL_WIDTH / GLOB_STORE_WIDTH;   // 12 / 2 = 6
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
    }

    // reg -> global
    #pragma unroll
    for (int i=0; i<GLOB_STORE_NUM; i++) {
      VecCpy<GLOB_STORE_WIDTH>(&C[(by * BM + sh_load_row) * N + 
                                  bx * BN + i * GLOB_STORE_ROW_WIDTH + sh_load_col * GLOB_STORE_WIDTH], 
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
  const int GLOB_STORE_WIDTH
>
__global__ void  __launch_bounds__(512) matmul_noadjust(float* A, float* B, float* C, const int M, const int N, const int K) {
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

  // prefetch load glob to shared
  #pragma unroll
  for (int i=0; i<GLOB_LOAD_NUM_A; i++) {
    VecCpy<GLOB_LOAD_WIDTH_A>(&shA[sh_load_row * BM + 
                                   i * GLOB_LOAD_ROW_WIDTH_A + 
                                   sh_load_col * GLOB_LOAD_WIDTH_A], 
                                &A[sh_load_row * M + 
                                   i * GLOB_LOAD_ROW_WIDTH_A + 
                                   sh_load_col * GLOB_LOAD_WIDTH_A]);
  }
  #pragma unroll
  for (int i=0; i<GLOB_LOAD_NUM_B; i++) {
    VecCpy<GLOB_LOAD_WIDTH_B>(&shB[sh_load_row * BN + 
                                   i * GLOB_LOAD_ROW_WIDTH_B + 
                                   sh_load_col * GLOB_LOAD_WIDTH_B], 
                                &B[sh_load_row * N + 
                                   i * GLOB_LOAD_ROW_WIDTH_B + 
                                   sh_load_col * GLOB_LOAD_WIDTH_B]);
  }
  __syncthreads();

  // for k
  int write_buffer_id = 1;
  for (int k=BK; k<=K; k+=BK) {
    if (k < K) {  // 最后一轮迭代只有数据计算，没有glob -> shared的操作
      #pragma unroll
      for (int i=0; i<GLOB_LOAD_NUM_A; i++) {
        VecCpy<GLOB_LOAD_WIDTH_A>(&tempA[i * GLOB_LOAD_WIDTH_A], 
                                      &A[(sh_load_row + k) * M + 
                                         i * GLOB_LOAD_ROW_WIDTH_A + 
                                         sh_load_col * GLOB_LOAD_WIDTH_A]);
      }
      #pragma unroll
      for (int i=0; i<GLOB_LOAD_NUM_B; i++) {
        VecCpy<GLOB_LOAD_WIDTH_B>(&tempB[i * GLOB_LOAD_WIDTH_B], 
                                    &B[(sh_load_row + k) * N + 
                                       i * GLOB_LOAD_ROW_WIDTH_B + 
                                       sh_load_col * GLOB_LOAD_WIDTH_B]);
      }
    }
    // for bk
    // prefetch load shared to reg
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

      // calculate result
      #pragma unroll
      for (int cy=0; cy<TM; cy++) {
        #pragma unroll
        for (int cx=0; cx<TN; cx++) {
          regC[cy * TN + cx] += regA[(bk % 2) * TM + cy] * regB[(bk % 2) * TN + cx];
        }
      }
    }

    // last calculate result
    #pragma unroll
    for (int cy=0; cy<TM; cy++) {
      #pragma unroll
      for (int cx=0; cx<TN; cx++) {
        regC[cy * TN + cx] += regA[TM + cy] * regB[TN + cx];
      }
    }

    if (k < K) {  // 最后只有数据计算
      #pragma unroll
      for (int i=0; i<GLOB_LOAD_NUM_A; i++) {
        VecCpy<GLOB_LOAD_WIDTH_A>(&shA[write_buffer_id * BM * BK + 
                                       sh_load_row * BM + 
                                       i * GLOB_LOAD_ROW_WIDTH_A + 
                                       sh_load_col * GLOB_LOAD_WIDTH_A], 
                                &tempA[i * GLOB_LOAD_WIDTH_A]);
      }
      #pragma unroll
      for (int i=0; i<GLOB_LOAD_NUM_B; i++) {
        VecCpy<GLOB_LOAD_WIDTH_B>(&shB[write_buffer_id * BN * BK + 
                                       sh_load_row * BN + 
                                       i * GLOB_LOAD_ROW_WIDTH_B + 
                                       sh_load_col * GLOB_LOAD_WIDTH_B], 
                                &tempB[i * GLOB_LOAD_WIDTH_B]);
      }
      __syncthreads();
      write_buffer_id ^= 1;
    }
  }

  if (LOCAL_SPLIT_U > 1) {
    __syncthreads();
    const int LDS_C_STRIDE = BM * BN;
    const int GLOB_STORE_TOTAL_WIDTH = LDS_C_STRIDE / THREAD_NUM;   // 3072 / 256 = 12
    const int GLOB_STORE_NUM = GLOB_STORE_TOTAL_WIDTH / GLOB_STORE_WIDTH;   // 12 / 2 = 6
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
    }

    // reg -> global
    #pragma unroll
    for (int i=0; i<GLOB_STORE_NUM; i++) {
      VecCpy<GLOB_STORE_WIDTH>(&C[(by * BM + sh_load_row) * N + 
                                  bx * BN + i * GLOB_STORE_ROW_WIDTH + sh_load_col * GLOB_STORE_WIDTH], 
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

// cublas gemm
cublasStatus_t cublasMatMulTransA(cublasHandle_t handle, const float* A, const float* B, float* C, int M, int N, int K, bool isTranA, bool isTranB) {
    float alpha = 1.0f;
    float beta = 0.0f;
    cublasStatus_t status;
    bool handleCreated = false;
    if (!handle) {
        status = cublasCreate(&handle);
        if (status != CUBLAS_STATUS_SUCCESS) {
            return status;
        }
        handleCreated = true;
    }
    cublasOperation_t tranA, tranB;
    int lda, ldb;
    if (isTranA && isTranB) {
        tranA = CUBLAS_OP_N; tranB = CUBLAS_OP_N;
        lda = M; ldb = K;
    } else if (!isTranA && isTranB) {
        tranA = CUBLAS_OP_T; tranB = CUBLAS_OP_N;
        lda = K; ldb = K;
    } else if (isTranA && !isTranB) {
        tranA = CUBLAS_OP_N; tranB = CUBLAS_OP_T;
        lda = M; ldb = N;
    } else {
        tranA = CUBLAS_OP_T; tranB = CUBLAS_OP_T;
        lda = K; ldb = N;
    }
    status = cublasSgemm(handle,
                        tranA,   // A不转置
                        tranB,   // B转置
                        M,             // 结果矩阵行数
                        N,             // 结果矩阵列数
                        K,             // 公共维度
                        &alpha,
                        A, lda,          // A的维度M×K，lda=M
                        B, ldb,          // B的维度NxK，ldb=N
                        &beta,
                        C, M);         // C的维度M×N，ldc=M

    // 清理临时创建的句柄
    if (handleCreated) {
        cublasDestroy(handle);
    }
    return status;
}

// gpu 验证
__global__ void verify_kernel(float* C, float* D, int m, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < n) {
      float sub = C[row * n + col] - D[col * m + row];
      if (sub >= 0.0000001f || sub <= -0.0000001f) {
        // printf("%d %d\n", row, col);
        printf("error!\nindex: (y=%d, x=%d)\nmine: %f  verify: %.8f\nsub: %.8f\n", row, col, C[row * n + col], D[col * m + row], sub);
      }
    }
}

int main() {
  int device_count;
  cudaGetDeviceCount(&device_count);
  int device_id;
  cudaGetDevice(&device_id);
  cudaSetDevice(device_id);

  const int M = 1024;
  const int N = 1024;
  const int K = 1024;

  const int BM = 64;
  const int BN = 64;
  const int BK = 32;
  const int TM = 4;
  const int TN = 4;

  const int GLOB_LOAD_WIDTH_A = 4;
  const int GLOB_LOAD_WIDTH_B = 4;

  const int BLOCK_LAYOUT_M = 2;   // BM / TM / WARP_LAYOUT_M
  const int BLOCK_LAYOUT_N = 4;    // BN / TN / WARP_LAYOUT_N
  const int WARP_LAYOUT_M = 8;
  const int WARP_LAYOUT_N = 4;

  const int WARP_SCATTER_WIDTH_A = 2;
  const int WARP_SCATTER_WIDTH_B = 2;
  const int THREAD_SCATTER_WIDTH_A = 1;
  const int THREAD_SCATTER_WIDTH_B = 1;

  const int LOCAL_SPLIT_U = 1;   /*2*/
  const int BLOCK_MAPPING = 8;
  const int WARP_SIZE = 32;
  const int GLOB_STORE_WIDTH = 4;

  float *A = new float[M * K];
  float *B = new float[N * K];
  float *C = new float[N * M];
  float *D = new float[N * M];
  for (int i = 0; i < M * N; i++) {
    C[i] = 0.0f;
    D[i] = 0.0f;
  }
  for (int i = 0; i < M * K; i++) {
    if (i % 2) A[i] = 0.0f;
    else A[i] = 1.0f;
    // A[i] = (rand() % 1000) * 0.01f;
  } 
  for (int i = 0; i < N * K; i++) {
    if (i % 2) B[i] = 0.0f;
    else B[i] = 1.0f;
    // B[i] = (rand() % 1000) * 0.01f;
  }

  float *DA, *DB, *DC, *DD;
  cudaMalloc(&DA, M * K * sizeof(float));
  cudaMalloc(&DB, N * K * sizeof(float));
  cudaMalloc(&DC, N * M * sizeof(float));
  cudaMalloc(&DD, N * M * sizeof(float));
  cudaMemcpy(DA, A, M * K * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(DB, B, N * K * sizeof(float), cudaMemcpyHostToDevice);
  
  dim3 grid_size((M/BM)*(N/BN));  // 1024/64=16; 1056/48=22
  dim3 block_size(((BM/TM)*(BN/TN)) * LOCAL_SPLIT_U);  // 64/4=16; 48/6=8

  dim3 dimBlock = {16, 16};
  dim3 dimGrid = {(N + dimBlock.x - 1) / dimBlock.x, (M + dimBlock.y - 1) / dimBlock.y};


  std::vector<float> costs;
  for (int i=0; i<1; i++) {
    // 执行内核函数
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    cudaEventRecord(startEvent, 0);

    // matmul_adjust<BM, BN, BK, TM, TN, 
    //   GLOB_LOAD_WIDTH_A, GLOB_LOAD_WIDTH_B,
    //   BLOCK_LAYOUT_M, BLOCK_LAYOUT_N,
    //   WARP_LAYOUT_M, WARP_LAYOUT_N, 
    //   WARP_SCATTER_WIDTH_A, WARP_SCATTER_WIDTH_B, 
    //   THREAD_SCATTER_WIDTH_A, THREAD_SCATTER_WIDTH_B,
    //   LOCAL_SPLIT_U, BLOCK_MAPPING, WARP_SIZE, GLOB_STORE_WIDTH><<<grid_size, block_size>>>(DA, DB, DC, M, N, K);
  
    matmul_noadjust<BM, BN, BK, TM, TN, 
      GLOB_LOAD_WIDTH_A, GLOB_LOAD_WIDTH_B,
      BLOCK_LAYOUT_M, BLOCK_LAYOUT_N,
      WARP_LAYOUT_M, WARP_LAYOUT_N, 
      WARP_SCATTER_WIDTH_A, WARP_SCATTER_WIDTH_B, 
      THREAD_SCATTER_WIDTH_A, THREAD_SCATTER_WIDTH_B,
      LOCAL_SPLIT_U, BLOCK_MAPPING, WARP_SIZE, GLOB_STORE_WIDTH><<<grid_size, block_size>>>(DA, DB, DC, M, N, K);
    
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasMatMulTransA(handle, DA, DB, DD, M, N, K, true, false);
    verify_kernel<<<dimGrid, dimBlock>>>(DC, DD, M, N);

    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
    costs.push_back(elapsedTime);
  }

  std::sort(costs.begin(), costs.end());
  float time = costs[costs.size()/2];
  double tflops = (2 * static_cast<uint64_t>(M) * N * K) / (time / 1000) / 1e12;

  std::cout << "time cost: " << time << "ms\n";
  std::cout << "tflops: " << tflops << std::endl;

  cudaFree(DA);
  cudaFree(DB);
  cudaFree(DC);
  cudaFree(DD);
  delete[] A;
  delete[] B;
  delete[] C;
  delete[] D;
  return 0;
}