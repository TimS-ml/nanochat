"""
Tokenizing Distributed Data Loader for Pre-training

This module provides a streaming data loader that:
1. Reads documents from Parquet files
2. Tokenizes them on-the-fly using multithreading
3. Yields batches of tokens for training
4. Supports approximate resumption from checkpoints
5. Works efficiently with distributed training (DDP)

Key design decisions:
- Streaming: Doesn't load entire dataset into memory
- On-the-fly tokenization: Amortizes tokenization cost over training
- Stateful: Can save/resume position in dataset for checkpoint recovery
- Distributed: Each rank processes different parts of the data
"""
from collections import deque

import torch
import pyarrow.parquet as pq

from nanochat.common import get_dist_info
from nanochat.dataset import list_parquet_files
from nanochat.tokenizer import get_tokenizer

def tokenizing_distributed_data_loader_with_state(B, T, split, tokenizer_threads=4, tokenizer_batch_size=128, device="cuda", resume_state_dict=None):
    """
    Stream pre-training data with on-the-fly tokenization and stateful resumption.

    This is a generator function that yields batches of tokenized data for training.
    It supports approximate checkpoint resumption, which is useful for:
    - Recovering from crashes during long training runs
    - Continuing training after early stopping
    - Resuming from checkpoints with loss spikes

    Resumption strategy:
    - APPROXIMATE, not perfect (trades complexity for simplicity)
    - Won't repeat the same documents
    - May skip a few documents at the resume point
    - Perfect resumption is possible but would require much more complex state tracking

    Args:
        B: Batch size (number of sequences)
        T: Sequence length (tokens per sequence)
        split: "train" or "val" - which dataset split to use
        tokenizer_threads: Number of threads for parallel tokenization (default: 4)
        tokenizer_batch_size: Batch size for tokenization (default: 128)
        device: Device to place tensors on (default: "cuda")
        resume_state_dict: Optional state dict to resume from a previous position
                          Should contain {'pq_idx': int, 'rg_idx': int}

    Yields:
        Tuple of (inputs, targets, state_dict):
        - inputs: Tensor of shape (B, T) with input token IDs
        - targets: Tensor of shape (B, T) with target token IDs (shifted by 1)
        - state_dict: Current position in dataset for checkpoint saving

    Example:
        >>> loader = tokenizing_distributed_data_loader_with_state(4, 512, "train")
        >>> for inputs, targets, state in loader:
        ...     loss = model(inputs, targets)
        ...     # Save state with checkpoint for resumption
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"

    # Get distributed training info
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()

    def document_batches():
        """
        Infinite generator that yields batches of documents (text strings).

        Handles:
        - Multi-epoch iteration (loops infinitely over dataset)
        - Distributed data parallel (each rank gets different row groups)
        - Resumption from a specific parquet file and row group
        """
        parquet_paths = list_parquet_files()
        # Split dataset: all but last file for train, last file for validation
        parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]

        # Determine starting position (for resumption or fresh start)
        resume_pq_idx = resume_state_dict["pq_idx"] if resume_state_dict is not None else 0
        resume_rg_idx = resume_state_dict["rg_idx"] if resume_state_dict is not None else None
        pq_idx = resume_pq_idx  # Start from resume point or beginning

        while True:  # Infinite multi-epoch iteration
            while pq_idx < len(parquet_paths):  # Iterate over parquet files
                filepath = parquet_paths[pq_idx]
                pf = pq.ParquetFile(filepath)

                # Determine starting row group index
                # This logic is a bit tricky due to approximate resumption + DDP
                if resume_rg_idx is not None:
                    # Resuming from a checkpoint
                    # Convert row group index to base units (chunks of world_size)
                    base_idx = resume_rg_idx // ddp_world_size
                    # Advance by 1 to avoid repeating data
                    base_idx += 1
                    # Each rank processes different row groups (strided by world_size)
                    rg_idx = base_idx * ddp_world_size + ddp_rank
                    # Clear resume flag (only applies to first file)
                    resume_rg_idx = None
                else:
                    # Fresh start: each rank starts at its rank offset
                    rg_idx = ddp_rank

                # Process row groups assigned to this rank
                while rg_idx < pf.num_row_groups:
                    rg = pf.read_row_group(rg_idx)
                    # Extract text documents from the 'text' column
                    batch = rg.column('text').to_pylist()

                    # Yield documents in smaller sub-batches for tokenization
                    # This allows better parallelization in the tokenizer
                    for i in range(0, len(batch), tokenizer_batch_size):
                        yield batch[i:i+tokenizer_batch_size], (pq_idx, rg_idx)

                    # Advance to next row group for this rank (strided by world_size)
                    rg_idx += ddp_world_size

                # Move to next parquet file
                pq_idx += 1

    batches = document_batches()

    # -------------------------------------------------------------------------
    # Token batching: accumulate tokens and yield training batches
    # -------------------------------------------------------------------------

    # Calculate how many tokens we need per batch
    needed_tokens = B * T + 1  # +1 because we need one extra for the shifted targets

    # Initialize tokenizer
    tokenizer = get_tokenizer()
    bos_token = tokenizer.get_bos_token_id()  # Beginning-of-sequence token

    # Token buffer: accumulates tokens from documents until we have enough for a batch
    token_buffer = deque()  # Efficient for append-right, pop-left operations

    while True:
        # Accumulate tokens until we have enough for one batch
        while len(token_buffer) < needed_tokens:
            # Get next batch of documents
            doc_batch, (pq_idx, rg_idx) = next(batches)

            # Tokenize the documents in parallel
            token_lists = tokenizer.encode(doc_batch, prepend=bos_token, num_threads=tokenizer_threads)

            # Add all tokens to the buffer
            for tokens in token_lists:
                token_buffer.extend(tokens)

        # Extract exactly the number of tokens we need
        tokens = [token_buffer.popleft() for _ in range(needed_tokens)]

        # Create tensor with optional CUDA optimizations
        use_cuda_optimizations = device == "cuda"
        scratch = torch.tensor(tokens, dtype=torch.long, pin_memory=use_cuda_optimizations)

        # Split into inputs and targets (targets are inputs shifted by 1)
        # This is standard for language modeling: predict the next token
        inputs_cpu = scratch[:-1]  # All tokens except the last
        targets_cpu = scratch[1:]  # All tokens except the first

        # Reshape from 1D to 2D (B, T) and transfer to device
        inputs = inputs_cpu.view(B, T).to(device=device, non_blocking=use_cuda_optimizations)
        targets = targets_cpu.view(B, T).to(device=device, non_blocking=use_cuda_optimizations)

        # Save current position for checkpoint resumption
        state_dict = {"pq_idx": pq_idx, "rg_idx": rg_idx}

        yield inputs, targets, state_dict

def tokenizing_distributed_data_loader(*args, **kwargs):
    """
    Wrapper around tokenizing_distributed_data_loader_with_state that only yields
    inputs and targets (without the state_dict).

    This is convenient when you don't need checkpoint resumption.

    Yields:
        Tuple of (inputs, targets) where both are tensors of shape (B, T)
    """
    for inputs, targets, state_dict in tokenizing_distributed_data_loader_with_state(*args, **kwargs):
        yield inputs, targets
