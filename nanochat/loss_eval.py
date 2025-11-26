"""
Evaluation functions for measuring language model quality using bits-per-byte metric.

The bits-per-byte (bpb) metric is superior to raw cross-entropy loss because it's
independent of tokenization and vocab size, allowing fair comparison across models
with different tokenizers.
"""
import math
import torch
import torch.distributed as dist

@torch.no_grad()
def evaluate_bpb(model, batches, steps, token_bytes):
    """
    Evaluate a model using bits-per-byte (bpb), a tokenization-independent metric.

    Instead of the naive 'mean loss', this function returns the bits per byte (bpb),
    which is a tokenization vocab size-independent metric, meaning you are still comparing
    apples:apples if you change the vocab size. The way this works is that instead of just
    calculating the average loss as usual, you calculate the sum loss, and independently
    also the sum bytes (of all the target tokens), and divide. This normalizes the loss by
    the number of bytes that the target tokens represent.

    Unlike raw cross-entropy loss, bits-per-byte normalizes by the actual number
    of bytes represented by the text, making it comparable across different
    tokenizers and vocabulary sizes. A model with vocab_size=10k and one with
    vocab_size=50k can be fairly compared using this metric.

    The metric works by:
    1. Computing the total cross-entropy loss (in nats) across all tokens
    2. Computing the total number of bytes represented by those tokens
    3. Converting to bits-per-byte: nats / (log(2) * bytes)

    Args:
        model: Language model with forward(x, y, loss_reduction='none') method
        batches: Iterator yielding (input_ids, target_ids) batches
        steps: Number of batches to evaluate on
        token_bytes: Tensor of shape (vocab_size,) mapping token IDs to byte counts
                     Special tokens should have value 0 to exclude them from the metric

    Returns:
        float: Bits-per-byte score (lower is better)
               Returns inf if no valid tokens were processed

    Exclusions from the metric:
        - Special tokens (token_bytes[id] == 0)
        - Masked tokens (target_ids == -1 or other negative values)
        - These are ignored because they don't represent actual text content

    Example:
        >>> token_bytes = torch.tensor([0, 1, 1, 2, 3, ...])  # BOS=0, then byte counts
        >>> bpb = evaluate_bpb(model, val_loader, steps=100, token_bytes=token_bytes)
        >>> print(f"Validation bpb: {bpb:.3f}")
        Validation bpb: 1.234

    Note:
        This function supports distributed evaluation. In multi-GPU settings,
        the total_nats and total_bytes are aggregated across all processes
        before computing the final metric.
    """
    # Accumulators for total loss (in nats) and total bytes
    # Use appropriate devices to match the model
    total_nats = torch.tensor(0.0, dtype=torch.float32, device=model.get_device())
    total_bytes = torch.tensor(0, dtype=torch.int64, device=model.get_device())

    # Iterate through batches
    batch_iter = iter(batches)
    for _ in range(steps):
        x, y = next(batch_iter)
        # Get per-token losses without reduction
        loss2d = model(x, y, loss_reduction='none')  # (B, T)
        # Flatten for easier processing
        loss2d = loss2d.view(-1)  # (B*T,)
        y = y.view(-1)  # (B*T,)

        # Handle negative target indices (masked tokens that should be ignored)
        if (y.int() < 0).any():
            # Note: MPS backend doesn't support < 0 for int64, only int32
            # Complex path: some targets are masked (e.g. -1 from ignore_index)
            # We need to avoid indexing token_bytes with negative indices
            valid = y >= 0
            # Create a safe version of y with no negatives (use 0 as placeholder)
            y_safe = torch.where(valid, y, torch.zeros_like(y))
            # Map valid targets to their byte length; masked targets get 0 bytes
            num_bytes2d = torch.where(
                valid,
                token_bytes[y_safe],
                torch.zeros_like(y, dtype=token_bytes.dtype)
            )
            # Only accumulate loss for tokens with non-zero byte count
            # This excludes both masked tokens and special tokens
            total_nats += (loss2d * (num_bytes2d > 0)).sum()
            total_bytes += num_bytes2d.sum()
        else:
            # Fast path: no masked targets, safe to index directly
            num_bytes2d = token_bytes[y]
            # Only accumulate loss for tokens with non-zero byte count
            # This excludes special tokens (which have token_bytes[id] = 0)
            total_nats += (loss2d * (num_bytes2d > 0)).sum()
            total_bytes += num_bytes2d.sum()

    # Aggregate across all processes if running distributed
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    if world_size > 1:
        # Sum total_nats and total_bytes across all ranks
        # After this, all ranks have the same global totals
        dist.all_reduce(total_nats, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_bytes, op=dist.ReduceOp.SUM)

    # Convert to Python scalars for final calculation
    total_nats = total_nats.item()
    total_bytes = total_bytes.item()

    # Avoid division by zero if no valid tokens were processed
    if total_bytes == 0:
        return float('inf')

    # Convert from nats-per-byte to bits-per-byte
    # - Cross-entropy loss is in nats (natural log base e)
    # - To convert to bits, divide by log(2)
    # - bpb = nats/bytes / log(2) = nats / (log(2) * bytes)
    bpb = total_nats / (math.log(2) * total_bytes)
    return bpb
