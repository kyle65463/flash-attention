import torch
import numpy as np

def python_attention(q, k, v):
    batch, heads, seq_len, dim = q.shape
    device, dtype = q.device, q.dtype
    
    q_np = q.detach().cpu().numpy()
    k_np = k.detach().cpu().numpy()
    v_np = v.detach().cpu().numpy()
    
    output = np.zeros_like(q_np)
    
    for b in range(batch):
        for h in range(heads):
            scores = np.zeros((seq_len, seq_len))
            for i in range(seq_len):
                for j in range(seq_len):
                    scores[i, j] = np.dot(q_np[b, h, i], k_np[b, h, j]) / np.sqrt(dim)
            
            for i in range(seq_len):
                max_val = np.max(scores[i])
                exp_scores = np.exp(scores[i] - max_val)
                sum_exp = np.sum(exp_scores)
                
                for d in range(dim):
                    weighted_sum = 0
                    for j in range(seq_len):
                        weighted_sum += (exp_scores[j] / sum_exp) * v_np[b, h, j, d]
                    output[b, h, i, d] = weighted_sum
    
    return torch.tensor(output, device=device, dtype=dtype)

def pytorch_attention(q, k, v):
    return torch.nn.functional.scaled_dot_product_attention(q, k, v)