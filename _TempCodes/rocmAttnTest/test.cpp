__global__ void attention(float* Q, float* K, float* V, float* O) {
  const int warp_id = threadIdx.x / 64;
  const int lane_id = threadIdx.x % 64;
  const int warp_y = warp_id / 1;
  const int warp_x = warp_id % 1;
  const int lane_y = lane_id / 16;
  const int lane_x = lane_id % 16;
  const int warp_y_ = warp_id / 2;
  const int warp_x_ = warp_id % 2;
  const int lane_y_ = lane_id / 8;
  const int lane_x_ = lane_id % 8;
  __shared__ float LDSMemory[4704];
  float* smQ = (float *)(LDSMemory);
  float* smK = (float *)(smQ + (32 * 16));
  float* smP = (float *)(smK + (64 * 16));
  float* smV = (float *)(smP + (32 * 64));
  float* smMax = (float *)(smV + (128 * 8));
  float* smSum = (float *)(smMax + (32));
  float* smFactor = (float *)(smSum + (32));
  #pragma unroll
  for (int i=0; i<32; i+=128) {
    if (i + threadIdx.x < 32) {
      smSum[i + threadIdx.x] = 0.0f;
      smMax[i + threadIdx.x] = -FLT_MAX;
    }
  }
  float tileO[4 * 8] = {0.0f};
  Q = Q + blockIdx.z * 32 * 2048 * 128 + blockIdx.y * 2048 * 128;
  K = K + blockIdx.z * 32 * 2048 * 128 + blockIdx.y * 2048 * 128;
  V = V + blockIdx.z * 32 * 2048 * 128 + blockIdx.y * 2048 * 128;
  O = O + blockIdx.z * 32 * 2048 * 128 + blockIdx.y * 2048 * 128;
  Q = Q + blockIdx.x * 32;
  O = O + blockIdx.x * 32 * 128;
  for (int bx=0; bx<2048; bx+=64) {
    float regQ[4], regK[4], tileP[4*4] = {0.0f};
    float regV[8], regP[4];
    float rowSum[4] = {0.0f};
    float rowMax[4] = {-FLT_MAX};
    for (int k=0; k<128; k+=16) {
      #pragma unroll
      for (int i=0; i<1; i++) {
        int x = i * 512 + threadIdx.x * 4;
        VecCpy<4>(&smQ[x], &Q[(x/32 + k) * 2048 + x%32]);
      }
      #pragma unroll
      for (int i=0; i<2; i++) {
        int x = i * 512 + threadIdx.x * 4;
        VecCpy<4>(&smK[x], &K[(x/64 + k) * 2048 + bx + x%64]);
      }
      __syncthreads();
      #pragma unroll
      for (int bk=0; bk<16; bk++) {
        #pragma unroll
        for (int i=0; i<1; i++) {
          #pragma unroll
          for (int j=0; j<1; j++) {
            int idx = (i * 2 + warp_y) * 4 * 4 + (j * 4 + lane_y) * 4;
            VecCpy<4>(&regQ[i * 4 + j * 4], &smQ[bk * 32 + idx]);
          }
        }
        #pragma unroll
        for (int i=0; i<2; i++) {
          #pragma unroll
          for (int j=0; j<1; j++) {
            int idx = (i * 1 + warp_x) * 16 * 2 + (j * 16 + lane_x) * 2;
            VecCpy<2>(&regK[i * 2 + j * 2], &smK[bk * 64 + idx]);
          }
        }
        #pragma unroll
        for (int cy=0; cy<4; cy++) {
          #pragma unroll
          for (int cx=0; cx<4; cx++) {
            tileP[cy * 4 + cx] += regQ[cy] * regK[cx] * 0.015625;
          }
        }
      }
      __syncthreads();
    }
    for (int i=0; i<4; i++) {
      for (int j=0; j<4; j++) {
        float oldMax = rowMax[i];
        rowMax[i] = fmaxf(oldMax, tileP[i * 4 + j]);
        rowSum[i] = rowSum[i] * __expf(oldMax - rowMax[i]) + __expf(tileP[i * 4 + j] - rowMax[i]);
      }
    }
    for (int i=0; i<4; i++) {
      for (int pos=1; pos<16; pos*=2) {
        float oldMax = __shfl_down(rowMax[i], pos, 16);
        float oldSum = __shfl_down(rowSum[i], pos, 16);
        float newMax = fmaxf(oldMax, rowMax[i]);
        rowSum[i] = oldSum * __expf(oldMax - newMax) + rowSum[i] * __expf(rowMax[i] - newMax);
        rowMax[i] = newMax;
      }
    }
    if (threadIdx.x % 16 == 0) {
      for (int i=0; i<1; i++) {
        for (int j=0; j<1; j++) {
          for (int k=0; k<4; k++) {
            int idx = (i * 2 + warp_y) * 4 * 4 + (j * 4 + lane_y) * 4 + k;
            int ii = i * 4 + j * 4 + k;
            float oldMax = smMax[idx];
            float oldSum = smSum[idx];
            float newMax = fmaxf(oldMax, rowMax[ii]);
            float factor = __expf(oldMax - newMax);
            smMax[idx] = newMax;
            smFactor[idx] = factor;
            smSum[idx] = oldSum * factor + rowSum[ii] * __expf(rowMax[ii] - newMax);
            rowMax[ii] = newMax;
          }
        }
      }
    }
    #pragma unroll
    for (int i=0; i<4; i++) {
      rowMax[i] = __shfl(rowMax[i], 0, 16);
    }
    for (int i=0; i<4; i++) {
      for (int j=0; j<4; j++) {
        tileP[i * 4 + j] = __expf(tileP[i * 4 + j] - rowMax[i]);
      }
    }
    #pragma unroll
    for (int i0=0; i0<1; i0++) {
      #pragma unroll
      for (int i1=0; i1<2; i1++) {
        #pragma unroll
        for (int j0=0; j0<1; j0++) {
          #pragma unroll
          for (int j1=0; j1<1; j1++) {
            #pragma unroll 
            for (int k=0; k<4; k++) {
              VecCpy<2>(&smP[((i0 * 2 + warp_y) * 4 * 4 + (j0 * 4 + lane_y) * 4 + k) * 64 + 
                                              (i1 * 1 + warp_x) * 16 * 2 + (j1 * 16 + lane_x) * 2], 
                                        &tileP[(i0 * 4 + j0 * 4 + k) * 4 + 
                                              i1 * 2 + j1 * 2]);
            }
          }
        }
      }
    }
    __syncthreads();
    float rowFactor[4];
    for (int i=0; i<1; i++) {
      for (int j=0; j<1; j++) {
        int idx = (i * 1 + warp_y_) * 8 * 4 + (j * 8 + lane_y_) * 4;
        VecCpy<4>(&rowFactor[i * 4 + j * 4], &smFactor[idx]);
      }
    }
    for (int i=0; i<4; i++) {
      for (int j=0; j<8; j++) {
        tileO[i * 8 + j] *= rowFactor[i];
      }
    }
    for (int k=0; k<64; k+=8) {
      __syncthreads();
      #pragma unroll
      for (int i=0; i<2; i++) {
        int x = i * 512 + threadIdx.x * 4;
        VecCpy<4>(&smV[x], &V[(x/128 + bx + k) * 128 + x%128]);
      }
      __syncthreads();
      #pragma unroll
      for (int bk=0; bk<8; bk++) {
        #pragma unroll
        for (int i=0; i<1; i++) {
          #pragma unroll
          for (int j=0; j<1; j++) {
            #pragma unroll
            for (int kk=0; kk<4; kk++) {
              int idx = (i * 1 + warp_y_) * 8 * 4 + (j * 8 + lane_y_) * 4 + kk;
              VecCpy<1>(&regP[i * 4 + j * 4 + kk], &smP[idx * 64 + k + bk]);
            }
          }
        }
        #pragma unroll
        for (int i=0; i<2; i++) {
          #pragma unroll
          for (int j=0; j<1; j++) {
            int idx = (i * 2 + warp_x_) * 8 * 4 + (j * 8 + lane_x_) * 4;
            VecCpy<4>(&regV[i * 4 + j * 4], &smV[bk * 128 + idx]);
          }
        }
        #pragma unroll
        for (int cy=0; cy<4; cy++) {
          #pragma unroll
          for (int cx=0; cx<8; cx++) {
            tileO[cy * 8 + cx] += regP[cy] * regV[cx];
          }
        }
      }
    }
  }
  float rowSum_[4];
  for (int i=0; i<1; i++) {
    for (int j=0; j<1; j++) {
      int idx = (i * 1 + warp_y_) * 8 * 4 + (j * 8 + lane_y_) * 4;
      VecCpy<4>(&rowSum_[i * 4 + j * 4], &smSum[idx]);
    }
  }
  for (int i=0; i<4; i++) {
    for (int j=0; j<8; j++) {
      tileO[i * 8 + j] /= rowSum_[i];
    }
  }
  #pragma unroll
  for (int i0=0; i0<1; i0++) {
    #pragma unroll
    for (int i1=0; i1<2; i1++) {
      #pragma unroll
      for (int j0=0; j0<1; j0++) {
        #pragma unroll
        for (int j1=0; j1<1; j1++) {
          #pragma unroll 
          for (int kk=0; kk<4; kk++) {
            VecCpy<4>(&O[((i0 * 1 + warp_y_) * 8 * 4 + (j0 * 8 + lane_y_) * 4 + kk) * 128 + 
                                           (i1 * 2 + warp_x_) * 8 * 4 + (j1 * 8 + lane_x_) * 4], 
                                      &tileO[(i0 * 4 + j0 * 4 + kk) * 8 + 
                                              i1 * 4 + j1 * 4]);
          }
        }
      }
    }
  }
}