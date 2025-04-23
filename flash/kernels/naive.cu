#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void naive_forward(const float* __restrict__ Q,
                              const float* __restrict__ K,
                              const float* __restrict__ V,
                              float* __restrict__ O,
                              int L, int d) {
    int bnh = blockIdx.x;
    int i   = threadIdx.x;

    const float* q = Q + bnh * L * d;
    const float* k = K + bnh * L * d;
    const float* v = V + bnh * L * d;
    float*       o = O + bnh * L * d;

    extern __shared__ float buf[];

    if (i < L) {
        // compute dot‑products against all keys
        for (int j = 0; j < L; ++j) {
            float dot = 0.f;
            for (int t = 0; t < d; ++t)
                dot += q[i * d + t] * k[j * d + t];
            buf[j] = dot / sqrtf((float)d);
        }
        // softmax
        float maxv = -FLT_MAX;
        for (int j = 0; j < L; ++j) maxv = fmaxf(maxv, buf[j]);
        float sum = 0.f;
        for (int j = 0; j < L; ++j) sum += (buf[j] = __expf(buf[j] - maxv));

        // weighted value sum
        for (int t = 0; t < d; ++t) {
            float acc = 0.f;
            for (int j = 0; j < L; ++j)
                acc += (buf[j] / sum) * v[j * d + t];
            o[i * d + t] = acc;
        }
    }
}

// Python binding
torch::Tensor forward(torch::Tensor q, torch::Tensor k, torch::Tensor v) {
    const int B = q.size(0), H = q.size(1), L = q.size(2), d = q.size(3);
    auto o = torch::empty_like(q);
    const int block = 1;
    naive_forward<<<B * H, block, L * sizeof(float)>>> (
        q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
        o.data_ptr<float>(), L, d);
    return o;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Naïve attention forward");
}