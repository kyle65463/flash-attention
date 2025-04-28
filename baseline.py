import torch


def pytorch_attn_math(q, k, v):
    """PyTorch SDPA with math kernel."""
    with torch.backends.cuda.sdp_kernel(
        enable_math=True, enable_flash=False, enable_mem_efficient=False
    ):
        return torch.nn.functional.scaled_dot_product_attention(q, k, v)


def pytorch_attn_flash(q, k, v):
    """PyTorch SDPA with flash attention kernel."""
    with torch.backends.cuda.sdp_kernel(
        enable_math=False, enable_flash=True, enable_mem_efficient=False
    ):
        return torch.nn.functional.scaled_dot_product_attention(q, k, v)


def pytorch_attn_efficient(q, k, v):
    """PyTorch SDPA with memory-efficient attention kernel."""
    with torch.backends.cuda.sdp_kernel(
        enable_math=False, enable_flash=False, enable_mem_efficient=True
    ):
        return torch.nn.functional.scaled_dot_product_attention(q, k, v)


def pytorch_manual_attn(q, k, v):
    """Manual implementation of attention in PyTorch without using SDPA."""
    # q, k, v shape: [batch, heads, seq, dim]
    
    # Calculate attention scores: (Q)(K^T)
    scores = torch.matmul(q, k.transpose(-2, -1))
    
    # Scale by 1/sqrt(d_k)
    d_k = q.size(-1)
    scores = scores / torch.sqrt(torch.tensor(d_k, dtype=scores.dtype, device=scores.device))
    
    # Apply softmax
    attn_weights = torch.nn.functional.softmax(scores, dim=-1)
    
    # Apply attention weights to values
    output = torch.matmul(attn_weights, v)
    
    return output


def flashattn_lib_attn(q, k, v):
    """Attention using the official flash-attn library."""
    try:
        from flash_attn import flash_attn_func
        
        # flash_attn expects inputs in [batch, seq, heads, dim] format
        # So we need to transpose from [batch, heads, seq, dim]
        q_t = q.transpose(1, 2)
        k_t = k.transpose(1, 2)
        v_t = v.transpose(1, 2)
        
        # Call flash attention
        output = flash_attn_func(q_t, k_t, v_t, causal=False)
        
        # Transpose back to original format [batch, heads, seq, dim]
        return output.transpose(1, 2)
    except ImportError:
        raise ImportError("flash-attn library is not installed. Please install with: pip install flash-attn")
