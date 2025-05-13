// source: https://github.com/kilianhae/FlashAttention.C/blob/main/src/flashattention.cu
#include <torch/extension.h>
#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <sys/time.h>

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>

#define NEG_INFINITY __int_as_float(0xff800000)

#define d 64
#define B_r 32
#define B_c 32
#define BK 64

#define TM 4
#define TN 4
#define CACHE_Q 0

/* -------------------------------------------------- */
/* ------------------  DEVICE CODE  ----------------- */
/* -------------------------------------------------- */

__global__ void flash_tiled_coarse_fp16(__half *out,     // FP16 I/O
                                        float *out_l,    // still float, small
                                        const __half *K, // FP16 inputs
                                        const __half *Q,
                                        const __half *V,
                                        float scaling,
                                        int batch_stride,
                                        int T_r,
                                        int T_c,
                                        int seq_len)
{
    /* thread & block bookkeeping (unchanged) */
    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;
    int batch_offset = batch_stride * blockIdx.x;

    /* ---------------- SMEM ---------------- */
    __shared__ __half Q_i[B_r][BK];
    __shared__ __half K_j[B_c][BK + 1]; // +1 to kill bank conflicts
    __shared__ __half V_j[B_c][BK];
    __shared__ float S_i[B_r][B_c + 1]; // keep in fp32 for soft-max

    const int num_tiles = d / BK;

    const uint totalResultsBlocktile = B_r * B_c;
    const uint numThreadsBlocktile = totalResultsBlocktile / (TM * TN);
    const int threadId_flat = threadIdx.y * blockDim.x + threadIdx.x;

    const int threadCol = threadId_flat % (B_c / TN);
    const int threadRow = threadId_flat / (B_c / TN);

    float l_i[TM] = {0.f};
    float m_i[TM]; // max logits running
    float last_m[TM];

    /* tiny per-thread output scratch (kept fp32, cast back later) */
    float O_i[num_tiles * TN * TM] = {0.f};

#pragma unroll
    for (int i = 0; i < TM; ++i)
        m_i[i] = -INFINITY;

    /* helpers for coalesced loads */
    const uint strideK = numThreadsBlocktile / BK;
    const uint innerRowK = threadId_flat / BK;
    const uint innerColK = threadId_flat % BK;

    const uint innerRowQ = threadId_flat / d;
    const uint innerColQ = threadId_flat % d;
    const uint nr_loads = B_r * d / numThreadsBlocktile;

    /* ------------ load first tile of Q to SMEM ------------ */
    for (int t = 0; t < nr_loads; ++t)
    {
        /* linear index this thread will load inside the 32 × 64 tile */
        int lin = threadId_flat + t * numThreadsBlocktile; // 0 … B_r*d-1
        int q_row = lin / d;                               // 0 … B_r-1
        int q_col = lin % d;                               // 0 … d-1

        int id = (blockIdx.y * B_r + q_row) * d + q_col; // global index
        __half qh = (id < d * seq_len) ? Q[batch_offset + id]
                                       : __float2half(0.f);

        Q_i[q_row][q_col] = qh;
    }
    __syncthreads();

    /* --------------- MAIN TILE LOOPS --------------- */
    float regM[TM];
    float regN[TN];

    for (int j = 0; j < T_c; ++j)
    {
        float threadResults[TM * TN] = {0.f};

        for (int t = 0; t < num_tiles; ++t)
        {
            /* --- load K tile --- */
            for (int i = 0; i < B_r; i += strideK)
            {
                int id = (innerRowK + j * B_c) * d + i * d + innerColK + t * B_c;
                __half k_val = (id < d * seq_len) ? K[batch_offset + id] : __float2half(0.f);
                K_j[innerRowK + i][innerColK] = k_val;
            }
            __syncthreads();

/* --- GEMM (fp32 accumulate) --- */
#pragma unroll
            for (int dd = 0; dd < BK; ++dd)
            {
#pragma unroll
                for (uint ii = 0; ii < TM; ++ii)
                    regM[ii] = __half2float(Q_i[threadRow * TM + ii][dd]);

#pragma unroll
                for (uint jj = 0; jj < TN; ++jj)
                    regN[jj] = __half2float(K_j[threadCol * TN + jj][dd]);

#pragma unroll
                for (uint ii = 0; ii < TM; ++ii)
#pragma unroll
                    for (uint jj = 0; jj < TN; ++jj)
                        threadResults[ii * TN + jj] += regM[ii] * regN[jj];
            }
            __syncthreads();
        } // tile loop

/* --- store scaled scores to S_i --- */
#pragma unroll
        for (uint ii = 0; ii < TM; ++ii)
#pragma unroll
            for (uint jj = 0; jj < TN; ++jj)
                S_i[threadRow * TM + ii][threadCol * TN + jj] =
                    threadResults[ii * TN + jj] * scaling;

        __syncthreads();

/* --- row-wise softmax bookkeeping (fp32) --- */
#pragma unroll
        for (int ii = 0; ii < TM; ++ii)
        {
            last_m[ii] = m_i[ii];
            float m = m_i[ii];
#pragma unroll
            for (int jj = 0; jj < B_c; ++jj)
                m = fmaxf(m, S_i[threadRow * TM + ii][jj]);
            m_i[ii] = m;
        }

        /* --- renormalise previous O & l --- */
        if (j > 0)
        {
            for (int t = 0; t < num_tiles; ++t)
#pragma unroll
                for (int ii = 0; ii < TM; ++ii)
                {
                    float factor = expf(last_m[ii] - m_i[ii]);
#pragma unroll
                    for (int jj = 0; jj < TN; ++jj)
                        O_i[t * TN * TM + ii * TN + jj] *= factor;
                }

#pragma unroll
            for (int ii = 0; ii < TM; ++ii)
                l_i[ii] *= expf(last_m[ii] - m_i[ii]);
        }

        /* --- process V & build new O chunk --- */
        for (int t = 0; t < num_tiles; ++t)
        {
            /* load V tile */
            __syncthreads();
            for (int i = 0; i < B_r; i += strideK)
            {
                int id = (innerRowK + j * B_c) * d + i * d + innerColK + t * B_c;
                __half v_val = (id < d * seq_len) ? V[batch_offset + id] : __float2half(0.f);
                V_j[innerRowK + i][innerColK] = v_val;
            }
            __syncthreads();

            for (int dd = 0; dd < B_c; ++dd)
            {
                float p[TM];
#pragma unroll
                for (int ii = 0; ii < TM; ++ii)
                {
                    p[ii] = expf(S_i[threadRow * TM + ii][dd] - m_i[ii]);
                    if (t == 0)
                        l_i[ii] += p[ii];
                }

#pragma unroll
                for (int jj = 0; jj < TN; ++jj)
                    regN[jj] = __half2float(V_j[dd][threadCol * TN + jj]);

#pragma unroll
                for (int ii = 0; ii < TM; ++ii)
#pragma unroll
                    for (int jj = 0; jj < TN; ++jj)
                        O_i[t * TN * TM + ii * TN + jj] += p[ii] * regN[jj];
            }
            __syncthreads();
        }
    } // chunk loop

    /* --- write normalised O back in FP16 --- */
    for (int t = 0; t < num_tiles; ++t)
#pragma unroll
        for (int ii = 0; ii < TM; ++ii)
        {
            float norm = 1.f / l_i[ii];
#pragma unroll
            for (int jj = 0; jj < TN; ++jj)
            {
                int row = blockIdx.y * B_r + threadRow * TM + ii;
                int col = t * B_c + threadCol * TN + jj;
                if (row < seq_len)
                {
                    int idx = batch_offset + row * d + col;
                    out[idx] = __float2half(O_i[t * TN * TM + ii * TN + jj] * norm);
                }
            }
        }
}

/* -------------------------------------------------- */
/* ------------------  HOST  API  -------------------- */
/* -------------------------------------------------- */

void run_flash_tiled_coarse(torch::Tensor &O,
                            torch::Tensor &O_l,
                            const torch::Tensor &K_d,
                            const torch::Tensor &Q_d,
                            const torch::Tensor &V_d,
                            int virtual_batch,
                            int seq_len)
{
    dim3 blockDim(B_r / TN, B_c / TM);
    dim3 gridDim(virtual_batch, (seq_len + B_r - 1) / B_r);

    flash_tiled_coarse_fp16<<<gridDim, blockDim>>>(
        reinterpret_cast<__half *>(O.data_ptr<at::Half>()),
        O_l.data_ptr<float>(),
        reinterpret_cast<const __half *>(K_d.data_ptr<at::Half>()),
        reinterpret_cast<const __half *>(Q_d.data_ptr<at::Half>()),
        reinterpret_cast<const __half *>(V_d.data_ptr<at::Half>()),
        1.f / sqrtf((float)d),
        seq_len * d,
        (seq_len + B_r - 1) / B_r,
        (seq_len + B_c - 1) / B_c,
        seq_len);

    cudaDeviceSynchronize();
}

/* thin wrapper identical to before, but expect half tensors */
torch::Tensor forward(torch::Tensor Q_d,
                      torch::Tensor K_d,
                      torch::Tensor V_d)
{
    TORCH_CHECK(Q_d.is_cuda() && K_d.is_cuda() && V_d.is_cuda(),
                "All tensors must be CUDA");
    TORCH_CHECK(Q_d.dtype() == at::kHalf,
                "Tensors must be at::Half (FP16)");

    TORCH_CHECK(Q_d.dim() == 4, "Q must be (B,H,S,D)");
    int B = Q_d.size(0);
    int H = Q_d.size(1);
    int S = Q_d.size(2);
    TORCH_CHECK(Q_d.size(3) == d,
                "Last dimension must be ", d);

    int virtual_batch = B * H;

    auto Qv = Q_d.contiguous().view({virtual_batch, S, d});
    auto Kv = K_d.contiguous().view({virtual_batch, S, d});
    auto Vv = V_d.contiguous().view({virtual_batch, S, d});

    auto O = torch::zeros({virtual_batch, S, d},
                          Q_d.options());
    auto O_l = torch::zeros({virtual_batch, S},
                            torch::dtype(at::kFloat).device(Q_d.device()));

    run_flash_tiled_coarse(O, O_l, Kv, Qv, Vv,
                           virtual_batch, S);

    return O.view({B, H, S, d});
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &forward,
          "Flash-tiled coarse attention (FP16)");
}
