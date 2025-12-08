from collections.abc import Iterable, Iterator
import json
import regex as re
# multiprocessing
import multiprocessing
import pathlib
import torch
from tqdm import tqdm
import math 

class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        """
        in_features: int final dimension of the input
        out_features: int final dimension of the output
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        self.std_dev = math.sqrt(2 / (in_features + out_features))
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=self.std_dev, a=-3*self.std_dev, b=3*self.std_dev)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the linear transformation to the input.
        """
        return x @ self.weight.T
    
class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        """
        num_embeddings: int Size of the vocabulary
        embedding_dim: int Dimension of the embedding vectors, i.e., dmodel
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = torch.nn.Parameter(torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype))
        self.std_dev = 1.0
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=self.std_dev, a=-3*self.std_dev, b=3*self.std_dev)
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Lookup the embedding vectors for the given token IDs.
        """
        return self.weight[token_ids]

class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        """
        d_model: int Dimension of the input
        eps: float = 1e-5 Small constant for numerical stability
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.scale = torch.nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMSNorm to the input.
        """
        assert x.shape[-1] == self.d_model, "Input last dimension must match d_model"
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        x_normalized = x / rms
        return (x_normalized * self.scale).to(in_dtype)
    

"""
Activation Functions and Feed-Forward Networks
"""

def silu(x: torch.Tensor) -> torch.Tensor:
    """
    Apply the SiLU activation function to the input.
    x: torch.Tensor Input tensor
    Returns:
        torch.Tensor Output tensor after applying SiLU
    """
    return x * torch.sigmoid(x)


class PositionwiseFeedForward(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)
        if d_ff is None:
            dff = (8 * d_model) // 3
            dff = 64*(dff//64) # Ensure dff is a multiple of 64 for hardware efficiency
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(silu(self.w1(x)) * self.w3(x))


class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None) -> None:
        """
        theta: float Base frequency for the rotary embeddings
        d_k: int Dimension of the input embeddings (must be even)
        max_seq_len: int Maximum sequence length to precompute embeddings for
        device: torch.device | None = None Device to store the precomputed embeddings on
        """
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device).float() / d_k))
        position = torch.arange(0, max_seq_len, device=device).float()
        sinusoid_inp = torch.einsum("i,j->ij", position, inv_freq)
        self.register_buffer("cos_cached", torch.cos(sinusoid_inp), persistent=False)
        self.register_buffer("sin_cached", torch.sin(sinusoid_inp), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Apply rotary positional embeddings to the input tensor at specified token positions.
        x: torch.Tensor Input tensor of shape (..., seq_len, d_k)
        token_positions: torch.Tensor Tensor of token positions to apply RoPE to, shape (batch_size, num_positions)
        Returns:
            torch.Tensor Output tensor of shape (..., seq_len, d_k) with RoPE applied at the specified positions
        """
        original_dtype = x.dtype
        x = x.to(torch.float32)
        sliced_x = x[..., token_positions, :]

        token_positions = token_positions.to(x.device)

        cos = self.cos_cached[token_positions] # shape (token_positions, d_k/2)
        sin = self.sin_cached[token_positions] # shape (token_positions, d_k/2)

        x1 = sliced_x[..., ::2] # even indices shape (..., token_positions, d_k/2)
        x2 = sliced_x[..., 1::2] # odd indices shape (..., token_positions, d_k/2)

        # element-wise multiplication
        rot_x1 = x1 * cos - x2 * sin
        rot_x2 = x1 * sin + x2 * cos

        sliced_x[..., ::2] = rot_x1
        sliced_x[..., 1::2] = rot_x2    

        x[..., token_positions, :] = sliced_x

        return x.to(original_dtype)

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Apply softmax to the input tensor along the specified dimension.
    x: torch.Tensor Input tensor
    dim: int Dimension along which to apply softmax
    Returns:
        torch.Tensor Output tensor after applying softmax
    """
    max_vals, _ = torch.max(x, dim=dim, keepdim=True)
    exp_x = torch.exp(x - max_vals)
    sum_exp_x = torch.sum(exp_x, dim=dim, keepdim=True)
    return exp_x / sum_exp_x


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Compute the scaled dot-product attention.
    query: torch.Tensor Query tensor of shape (..., seq_len_q, d_k)
    key: torch.Tensor Key tensor of shape (..., seq_len_k, d_k)
    value: torch.Tensor Value tensor of shape (..., seq_len_k, d_v)
    mask: torch.Tensor | None = None Optional boolean mask tensor of shape (..., seq_len_q, seq_len_k)
    Returns:
        torch.Tensor Output tensor of shape (..., seq_len_q, d_v)
    """
    d_k = query.shape[-1]
    scores = torch.einsum("...qd,...kd->...qk", query, key) / math.sqrt(d_k)

    if mask is not None:
        mask = mask.to(torch.bool)
        scores = scores.masked_fill(~mask, float('-inf'))

    attention = softmax(scores, dim=-1)
    return torch.einsum("...qk,...kv->...qv", attention, value)

"""
Implement causal multi-head self-attention as a torch.nn.Module. Your implemen-
tation should accept (at least) the following parameters:
d_model: int Dimensionality of the Transformer block inputs.
num_heads: int Number of heads to use in multi-head self-attention.
Folllowing Vaswani et al. [2017], set dk = dv = dmodel/h
Use causal masking in the attention computation.
"""
class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, theta: float | None = None, token_positions: torch.Tensor | None = None) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads

        self.query = Linear(d_model, self.d_k * num_heads)
        self.key = Linear(d_model, self.d_k * num_heads)
        self.value = Linear(d_model, self.d_v * num_heads)
        self.out = Linear(self.d_v * num_heads, d_model)

        self.theta = theta
        self.token_positions = token_positions


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()

        q = self.query(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_len, self.num_heads, self.d_v).transpose(1, 2)
   
        if self.theta is not None:
            if self.token_positions is None:
                self.token_positions = torch.arange(seq_len, device=x.device)
            rope = RotaryPositionalEmbedding(theta=self.theta, d_k=self.d_k, max_seq_len=seq_len, device=x.device)
            q = rope(q, self.token_positions)
            k = rope(k, self.token_positions)

        mask = torch.tril(torch.ones((seq_len, seq_len), device=x.device)).unsqueeze(0).unsqueeze(0)

        attention_output = scaled_dot_product_attention(q, k, v, mask)
        attention_output = attention_output.transpose(1,2).contiguous().view(batch_size, seq_len, self.num_heads * self.d_v)

        output = self.out(attention_output)
        return output

class TransformerBlock(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int) -> None:
        super().__init__()
        self.mha = MultiHeadSelfAttention(d_model, num_heads)
        self.rmsnorm1 = RMSNorm(d_model)
        self.ffn = PositionwiseFeedForward(d_model, d_ff)
        self.rmsnorm2 = RMSNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_output = self.mha(self.rmsnorm1(x))
        x = x + attn_output
        ffn_output = self.ffn(self.rmsnorm2(x))
        x = x + ffn_output
        return x
    
"""
Time to put it all together! Implement the Transformer language model as described in §3.1
and illustrated in Figure 1. At minimum, your implementation should accept all the aforementioned
construction parameters for the Transformer block, as well as these additional parameters:
vocab_size: embedding matrix.
int The size of the vocabulary, necessary for determining the dimensionality of the token
context_length: int The maximum context length, necessary for determining the dimensionality of
the position embedding matrix.
num_layers: int The number of Transformer blocks to use.
To test your implementation against our provided tests, you will first need to implement the test
adapter at [adapters.run_transformer_lm]. Then, run uv run pytest -k test_transformer_lm
to test your implementation.
Deliverable: A Transformer LM module that passes the above tests.
"""

class TransformerLM(torch.nn.Module):
    def __init__(self, vocab_size: int, context_length: int, d_model: int, num_heads: int, d_ff: int, num_layers: int) -> None:
        super().__init__()
        self.token_embedding = Embedding(vocab_size, d_model)
        self.layers = torch.nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff) for _ in range(num_layers)])
        self.rmsnorm = RMSNorm(d_model)
        self.output_linear = Linear(d_model, vocab_size)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(token_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.rmsnorm(x)
        logits = self.output_linear(x)
        return logits