#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>

__global__ void naive_attention_forward_kernel(const float* __restrict__ Q,
                                               const float* __restrict__ K,
                                               const float* __restrict__ V,
                                               float*       __restrict__ O,
                                               int B, int H, int S, int D) {
    int b = blockIdx.x;
    int h = blockIdx.y;
    int tid = threadIdx.x;
    float scale = rsqrtf(static_cast<float>(D));

    int head_stride  = S * D;
    int batch_stride = H * head_stride;
    int base = b * batch_stride + h * head_stride;

    for (int q = tid; q < S; q += blockDim.x) {
        const float* q_ptr = Q + base + q * D;

        float max_logit = -FLT_MAX;
        for (int k = 0; k < S; ++k) {
            const float* k_ptr = K + base + k * D;
            float dot = 0.f;
            for (int d = 0; d < D; ++d) dot += q_ptr[d] * k_ptr[d];
            float scaled = dot * scale;
            max_logit = fmaxf(max_logit, scaled);
        }

        float denom = 0.f;
        float* o_ptr = O + base + q * D;
        for (int d = 0; d < D; ++d) o_ptr[d] = 0.f;

        for (int k = 0; k < S; ++k) {
            const float* k_ptr = K + base + k * D;
            float dot = 0.f;
            for (int d = 0; d < D; ++d) dot += q_ptr[d] * k_ptr[d];
            float e = __expf(dot * scale - max_logit);
            denom += e;
            const float* v_ptr = V + base + k * D;
            for (int d = 0; d < D; ++d) o_ptr[d] += e * v_ptr[d];
        }

        float inv_denom = 1.f / denom;
        for (int d = 0; d < D; ++d) o_ptr[d] *= inv_denom;
    }
}

torch::Tensor forward(torch::Tensor q, torch::Tensor k, torch::Tensor v) {
    const int B = q.size(0);  // batch
    const int H = q.size(1);  // heads 
    const int S = q.size(2);  // sequence length
    const int D = q.size(3);  // embedding dimension

    auto o = torch::empty_like(q);
    
    auto q_float = q.to(torch::kFloat32);
    auto k_float = k.to(torch::kFloat32);
    auto v_float = v.to(torch::kFloat32);
    auto o_float = o.to(torch::kFloat32);
    
    const int threads = 128;
    dim3 grid(B, H);

    naive_attention_forward_kernel<<<grid, threads>>>(
        q_float.data_ptr<float>(),
        k_float.data_ptr<float>(),
        v_float.data_ptr<float>(),
        o_float.data_ptr<float>(),
        B, H, S, D
    );
    
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(error));
    }
    
    return o_float.to(q.dtype());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Naïve attention forward");
}