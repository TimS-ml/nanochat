"""
GPT model (rewrite, a lot simpler)
Notable features:
- rotary embeddings (and no positional embeddings)
- QK norm
- untied weights for token embedding and lm_head
- relu^2 activation in MLP
- norm after token embedding
- no learnable params in rmsnorm
- no bias in linear layers
- Multi-Query Attention (MQA) support for more efficient inference
"""

import math
from functools import partial
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import get_dist_info, print0
from nanochat.muon import Muon, DistMuon
from nanochat.adamw import DistAdamW

@dataclass
class GPTConfig:
    """
    Configuration class for GPT model architecture.

    This dataclass defines all the hyperparameters needed to instantiate a GPT model.
    Uses Multi-Query Attention (MQA) where n_kv_head can be less than n_head for efficiency.
    """
    sequence_len: int = 1024  # Maximum sequence length the model can process
    vocab_size: int = 50304   # Size of the vocabulary (number of unique tokens)
    n_layer: int = 12         # Number of transformer blocks/layers
    n_head: int = 6           # Number of query attention heads
    n_kv_head: int = 6        # Number of key/value heads (for Multi-Query Attention)
    n_embd: int = 768         # Embedding dimension size


def norm(x):
    """
    Apply RMS (Root Mean Square) normalization to the input tensor.
    This is a purely functional rmsnorm with no learnable parameters.

    RMS normalization normalizes the input by dividing by the root mean square of its values,
    which helps stabilize training and improve model performance.

    Args:
        x: Input tensor to normalize

    Returns:
        Normalized tensor with the same shape as input
    """
    # Purely functional rmsnorm with no learnable params
    return F.rms_norm(x, (x.size(-1),))


def apply_rotary_emb(x, cos, sin):
    """
    Apply Rotary Position Embeddings (RoPE) to the input tensor.

    RoPE encodes position information by rotating pairs of dimensions in the embedding space.
    This allows the model to capture relative positional information without absolute position embeddings.

    The rotation is applied by splitting the last dimension into two halves and applying a 2D rotation
    matrix to each pair of dimensions using the pre-computed cos and sin values.

    Args:
        x: Input tensor of shape (batch, seq_len, n_heads, head_dim)
        cos: Pre-computed cosine values for rotation
        sin: Pre-computed sine values for rotation

    Returns:
        Tensor with rotary embeddings applied, same shape as input
    """
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
    y1 = x1 * cos + x2 * sin # rotate pairs of dims using 2D rotation formula
    y2 = x1 * (-sin) + x2 * cos # complete the 2D rotation
    out = torch.cat([y1, y2], 3) # re-assemble the two halves
    out = out.to(x.dtype) # ensure input/output dtypes match
    return out

class CausalSelfAttention(nn.Module):
    """
    Causal (autoregressive) self-attention module with support for Multi-Query Attention (MQA).

    Features:
    - Rotary Position Embeddings (RoPE) for relative position encoding
    - QK normalization for training stability
    - Multi-Query Attention: fewer key/value heads than query heads for efficiency
    - KV caching support for fast autoregressive generation
    """
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx  # Layer index in the transformer stack
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        # Ensure embedding dimension is evenly divisible by number of heads
        assert self.n_embd % self.n_head == 0
        # Ensure MQA configuration is valid: n_kv_head must divide n_head evenly
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        # Query projection: maps embeddings to query vectors for all heads
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        # Key/Value projections: potentially fewer heads for MQA efficiency
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        # Output projection: maps attention output back to embedding dimension
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

    def forward(self, x, cos_sin, kv_cache):
        B, T, C = x.size()

        # Project the input to get queries, keys, and values
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Apply Rotary Embeddings to queries and keys to get relative positional encoding
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin) # QK rotary embedding
        q, k = norm(q), norm(k) # QK norm
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2) # make head be batch dim, i.e. (B, T, H, D) -> (B, H, T, D)

        # Apply KV cache: insert current k,v into cache, get the full view so far
        if kv_cache is not None:
            k, v = kv_cache.insert_kv(self.layer_idx, k, v)
        Tq = q.size(2) # number of queries in this forward pass
        Tk = k.size(2) # number of keys/values in total (in the cache + current forward pass)

        # Attention: queries attend to keys/values autoregressively. A few cases to handle:
        enable_gqa = self.n_head != self.n_kv_head # Group Query Attention (GQA): duplicate key/value heads to match query heads if desired
        if kv_cache is None or Tq == Tk:
            # During training (no KV cache), attend as usual with causal attention
            # And even if there is KV cache, we can still use this simple version when Tq == Tk
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa)
        elif Tq == 1:
            # During inference but with a single query in this forward pass:
            # The query has to attend to all the keys/values in the cache
            y = F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=enable_gqa)
        else:
            # During inference AND we have a chunk of queries in this forward pass:
            # First, each query attends to all the cached keys/values (i.e. full prefix)
            attn_mask = torch.zeros((Tq, Tk), dtype=torch.bool, device=q.device) # True = keep, False = mask
            prefix_len = Tk - Tq
            if prefix_len > 0: # can't be negative but could be zero
                attn_mask[:, :prefix_len] = True
            # Then, causal attention within this chunk
            attn_mask[:, prefix_len:] = torch.tril(torch.ones((Tq, Tq), dtype=torch.bool, device=q.device))
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, enable_gqa=enable_gqa)

        # Re-assemble the heads side by side and project back to residual stream
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (feedforward network) used in each transformer block.

    This implementation uses:
    - Standard 4x expansion of the embedding dimension
    - ReLU^2 activation (ReLU squared) instead of traditional GELU
    - No bias terms in linear layers for simplicity
    """
    def __init__(self, config):
        super().__init__()
        # First layer: expand from n_embd to 4*n_embd
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        # Second layer: project back down to n_embd
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)  # Expand dimension
        x = F.relu(x).square()  # Apply ReLU^2 activation (non-standard but effective)
        x = self.c_proj(x)  # Project back to original dimension
        return x


class Block(nn.Module):
    """
    A single transformer block consisting of attention and feedforward layers.

    Uses pre-normalization architecture (norm before attention/MLP) with residual connections.
    This is the "pre-LN" variant which has become standard in modern transformers.
    """
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, cos_sin, kv_cache):
        # Pre-norm + attention + residual connection
        x = x + self.attn(norm(x), cos_sin, kv_cache)
        # Pre-norm + MLP + residual connection
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    """
    GPT (Generative Pre-trained Transformer) language model.

    Architecture highlights:
    - Rotary position embeddings (RoPE) instead of learned absolute positions
    - QK normalization in attention for stability
    - Untied weights: separate embeddings for input tokens and output predictions
    - ReLU^2 activation in MLP layers
    - Pre-normalization (norm before attention/MLP)
    - No bias terms in linear layers
    - Multi-Query Attention (MQA) support for efficient inference
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),  # Token embeddings
            "h": nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)]),  # Transformer blocks
        })
        # Language model head: untied from input embeddings (not weight-shared)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Pre-compute rotary position embeddings
        # To support meta device initialization, we init the rotary embeddings here
        # We over-allocate by 10x to avoid recomputation, trading memory for convenience
        self.rotary_seq_len = config.sequence_len * 10  # 10x buffer for longer sequences
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        # Register as buffers (not parameters) and non-persistent (not saved in checkpoints)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def init_weights(self):
        """
        Initialize model weights using custom initialization strategy.

        This method:
        1. Initializes linear layers and embeddings using _init_weights helper
        2. Zeros out output projection weights (lm_head and residual projections)
        3. Re-initializes rotary embeddings on the correct device
        4. Converts token embeddings to bfloat16 to save memory
        """
        # Apply default initialization to all modules
        self.apply(self._init_weights)

        # Zero out classifier weights for better training dynamics
        torch.nn.init.zeros_(self.lm_head.weight)

        # Zero out residual projection weights in all transformer blocks
        # This helps with training stability at initialization
        for block in self.transformer.h:
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
            torch.nn.init.zeros_(block.attn.c_proj.weight)

        # Re-initialize the rotary embeddings on the correct device
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin

        # Cast token embeddings to bfloat16 for memory efficiency
        # The optimizer can handle this, and it saves memory in both model and activations
        if self.transformer.wte.weight.device.type == "cuda":
            self.transformer.wte.to(dtype=torch.bfloat16)

    def _init_weights(self, module):
        """
        Initialize individual module weights.

        For Linear layers: Uses a modified Xavier/Glorot initialization with aspect ratio correction
        Reference: https://arxiv.org/pdf/2310.17813

        For Embeddings: Standard normal initialization with std=1.0
        """
        if isinstance(module, nn.Linear):
            # Custom initialization that accounts for aspect ratio of the weight matrix
            fan_out = module.weight.size(0)
            fan_in = module.weight.size(1)
            # Standard deviation adjusted for matrix shape
            std = 1.0 / math.sqrt(fan_in) * min(1.0, math.sqrt(fan_out / fan_in))
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # Standard normal initialization for embeddings
            torch.nn.init.normal_(module.weight, mean=0.0, std=1.0)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        """
        Pre-compute cosine and sine values for Rotary Position Embeddings (RoPE).

        RoPE uses sinusoidal functions with different frequencies for each dimension pair,
        similar to the original Transformer positional encodings but applied as rotations.

        Args:
            seq_len: Maximum sequence length to pre-compute embeddings for
            head_dim: Dimension of each attention head
            base: Base for the geometric progression of frequencies (default 10000)
                  TODO: Consider using 100K which is more common in recent models
            device: Device to create tensors on (auto-detected if None)

        Returns:
            Tuple of (cos, sin) tensors with shape (1, seq_len, 1, head_dim//2)
        """
        # Auto-detect the device from model embeddings if not specified
        if device is None:
            device = self.transformer.wte.weight.device

        # Create frequency values for each dimension pair
        # We only need head_dim/2 frequencies since we rotate pairs of dimensions
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))

        # Create position indices for the sequence
        t = torch.arange(seq_len, dtype=torch.float32, device=device)

        # Calculate rotation angles at each (position, frequency) pair
        freqs = torch.outer(t, inv_freq)  # (seq_len, head_dim//2)

        # Pre-compute cos and sin for efficiency during forward pass
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16()  # Convert to bfloat16 to save memory

        # Add batch and head dimensions for broadcasting: (1, seq_len, 1, head_dim//2)
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin

    def get_device(self):
        """
        Get the device where the model is located.
        Uses the token embedding weight as a proxy for the entire model's device.
        """
        return self.transformer.wte.weight.device

    def estimate_flops(self):
        """
        Estimate the number of floating point operations (FLOPs) per token.

        This is useful for:
        - Comparing model efficiency
        - Estimating training/inference compute requirements
        - Model Flops Utilization (MFU) calculations

        Reference: https://arxiv.org/abs/2204.02311

        Returns:
            Estimated FLOPs per token
        """
        # Count total parameters and embedding parameters separately
        nparams = sum(p.numel() for p in self.parameters())
        nparams_embedding = self.transformer.wte.weight.numel()

        # Extract model dimensions
        l = self.config.n_layer  # number of layers
        h = self.config.n_head   # number of heads
        q = self.config.n_embd // self.config.n_head  # head dimension
        t = self.config.sequence_len  # sequence length

        # Calculate FLOPs: 6*params for forward pass + 12*l*h*q*t for attention
        num_flops_per_token = 6 * (nparams - nparams_embedding) + 12 * l * h * q * t
        return num_flops_per_token

    def setup_optimizers(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0):
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()
        # Separate out all parameters into 3 groups (matrix, embedding, lm_head)
        matrix_params = list(self.transformer.h.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        assert len(list(self.parameters())) == len(matrix_params) + len(embedding_params) + len(lm_head_params)
        # Create the AdamW optimizer for the embedding and lm_head
        # Scale the LR for the AdamW parameters by ∝1/√dmodel (having tuned the LRs for 768 dim model)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        if rank == 0:
            print(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")
        adam_groups = [
            dict(params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale),
            dict(params=embedding_params, lr=embedding_lr * dmodel_lr_scale),
        ]
        adamw_kwargs = dict(betas=(0.8, 0.95), eps=1e-10, weight_decay=weight_decay)
        AdamWFactory = DistAdamW if ddp else partial(torch.optim.AdamW, fused=True)
        adamw_optimizer = AdamWFactory(adam_groups, **adamw_kwargs)
        # Create the Muon optimizer for the linear layers
        muon_kwargs = dict(lr=matrix_lr, momentum=0.95)
        MuonFactory = DistMuon if ddp else Muon
        muon_optimizer = MuonFactory(matrix_params, **muon_kwargs)
        # Combine them the two optimizers into one list
        optimizers = [adamw_optimizer, muon_optimizer]
        for opt in optimizers:
            for group in opt.param_groups:
                group["initial_lr"] = group["lr"]
        return optimizers

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
        B, T = idx.size()

        # Grab the rotary embeddings for the current sequence length (they are of shape (1, seq_len, 1, head_dim))
        assert T <= self.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
        assert idx.device == self.cos.device, f"Rotary embeddings and idx are on different devices: {idx.device} != {self.cos.device}"
        assert self.cos.dtype == torch.bfloat16, "Rotary embeddings must be in bfloat16"
        # if kv cache exists, we need to offset the rotary embeddings to the current position in the cache
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T] # truncate cache to current sequence length

        # Forward the trunk of the Transformer
        x = self.transformer.wte(idx)
        x = norm(x)
        for block in self.transformer.h:
            x = block(x, cos_sin, kv_cache)
        x = norm(x)

        # Forward the lm_head (compute logits)
        softcap = 15
        if targets is not None:
            # training mode: compute and return the loss
            # TODO: experiment with Liger Kernels / chunked cross-entropy etc.
            logits = self.lm_head(x)
            logits = softcap * torch.tanh(logits / softcap) # logits softcap
            logits = logits.float() # use tf32/fp32 for logits
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
            return loss
        else:
            # inference mode: compute and return the logits
            logits = self.lm_head(x)
            logits = softcap * torch.tanh(logits / softcap) # logits softcap
            return logits

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """
        Naive autoregressive streaming inference.
        To make it super simple, let's assume:
        - batch size is 1
        - ids and the yielded tokens are simple Python lists and ints
        """
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device) # add batch dim
        for _ in range(max_tokens):
            logits = self.forward(ids) # (B, T, vocab_size)
            logits = logits[:, -1, :] # (B, vocab_size)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()
            yield token
