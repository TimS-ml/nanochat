"""
Tokenizer Training Script

This script trains a BPE (Byte Pair Encoding) tokenizer on the training data using
the HuggingFace tokenizers library. The tokenizer is trained in the style of GPT-4's
cl100k_base tokenizer.

BPE tokenization:
1. Start with all 256 possible bytes as the initial vocabulary
2. Iteratively merge the most frequent pair of tokens to create new tokens
3. Continue until reaching the target vocabulary size
4. The result is a vocabulary that efficiently compresses common patterns

The script also computes and caches token_bytes.pt, which maps each token ID to
the number of bytes it represents. This enables efficient calculation of bits per
byte (bpb) loss, which is invariant to vocabulary size.

Key parameters:
- vocab_size: Size of the final vocabulary (default 65536 = 2^16)
- max_chars: Maximum characters to train on (default 10B)
- doc_cap: Maximum characters per document (default 10K)

The tokenizer will be saved to <base_dir>/tokenizer/ with:
- tokenizer.json: Full tokenizer configuration
- special_tokens.json: Special token definitions
- token_bytes.pt: Tensor mapping token IDs to byte counts

Usage examples:

Default training (10B chars, vocab size 65536):
    python -m scripts.tok_train

Smaller vocabulary:
    python -m scripts.tok_train --vocab_size=32768

Train on less data (faster):
    python -m scripts.tok_train --max_chars=1000000000  # 1B chars

After training, use scripts.tok_eval to evaluate compression ratios.
"""
import os
import time
import argparse
import torch
from nanochat.tokenizer import RustBPETokenizer
from nanochat.common import get_base_dir
from nanochat.dataset import parquets_iter_batched

# -----------------------------------------------------------------------------
# Parse command line arguments

parser = argparse.ArgumentParser(description='Train a BPE tokenizer')
parser.add_argument('--max_chars', type=int, default=10_000_000_000, help='Maximum characters to train on (default: 10B)')
parser.add_argument('--doc_cap', type=int, default=10_000, help='Maximum characters per document (default: 10,000)')
parser.add_argument('--vocab_size', type=int, default=65536, help='Vocabulary size (default: 65536 = 2^16)')
args = parser.parse_args()
print(f"max_chars: {args.max_chars:,}")
print(f"doc_cap: {args.doc_cap:,}")
print(f"vocab_size: {args.vocab_size:,}")

# -----------------------------------------------------------------------------
# Text iterator

def text_iterator():
    """
    Create an iterator over training text for tokenizer training.

    This iterator:
    1. Loads documents from the training data in batches
    2. Crops each document to max doc_cap characters (prevents extremely long docs)
    3. Yields documents one at a time
    4. Stops after seeing max_chars total characters

    Cropping long documents is important because:
    - Very long documents can skew the token statistics
    - Training is more balanced when each document has similar weight
    - Speeds up training by limiting document size

    Yields:
        String documents from the training data
    """
    nchars = 0
    for batch in parquets_iter_batched(split="train"):
        for doc in batch:
            doc_text = doc
            if len(doc_text) > args.doc_cap:
                doc_text = doc_text[:args.doc_cap]
            nchars += len(doc_text)
            yield doc_text
            if nchars > args.max_chars:
                return
text_iter = text_iterator()

# -----------------------------------------------------------------------------
# Train the tokenizer
t0 = time.time()
tokenizer = RustBPETokenizer.train_from_iterator(text_iter, args.vocab_size)
t1 = time.time()
train_time = t1 - t0
print(f"Training time: {train_time:.2f}s")

# -----------------------------------------------------------------------------
# Save the tokenizer to disk
base_dir = get_base_dir()
tokenizer_dir = os.path.join(base_dir, "tokenizer")
tokenizer.save(tokenizer_dir)

# -----------------------------------------------------------------------------
# Quick inline sanity check
test_text = """Hello world! This is a test.
Numbers: 123, 4567, 89
Contractions: I'm, you're, it's
Special chars: @#$%^&*()
Unicode: 你好世界 🌍"""
encoded = tokenizer.encode(test_text)
decoded = tokenizer.decode(encoded)
assert decoded == test_text

# -----------------------------------------------------------------------------
# One more thing: we wish to cache a mapping from token id to number of bytes of that token
# for efficient evaluation of bits per byte. Unlike the typical mean loss, this
# allows us to report a loss that is invariant to the vocab size of the tokenizer.
# The bits per byte on the validation set is then one of the primary metrics we care about.
vocab_size = tokenizer.get_vocab_size()
special_set = set(tokenizer.get_special_tokens())
token_strings = [tokenizer.decode([token_id]) for token_id in range(vocab_size)]
token_bytes = []
for token_id in range(vocab_size):
    token_str = token_strings[token_id] # the Python string representation of this token
    if token_str in special_set:
        token_bytes.append(0) # special characters are not counted
    else:
        id_bytes = len(token_str.encode("utf-8")) # number of bytes that make up this token
        token_bytes.append(id_bytes)
token_bytes = torch.tensor(token_bytes, dtype=torch.int32, device='cpu')
token_bytes_path = os.path.join(tokenizer_dir, "token_bytes.pt")
with open(token_bytes_path, "wb") as f:
    torch.save(token_bytes, f)
print(f"Saved token_bytes to {token_bytes_path}")

# Log to report
from nanochat.report import get_report
token_bytes_nonzero = (token_bytes[token_bytes > 0]).to(dtype=torch.float32)
get_report().log(section="Tokenizer training", data=[
    vars(args), # argparse command line arguments
    {"train_time": train_time},
    {"num_special_tokens": len(special_set)},
    {
        "token_bytes_min": int(token_bytes_nonzero.min().item()),
        "token_bytes_max": int(token_bytes_nonzero.max().item()),
        "token_bytes_mean": token_bytes_nonzero.mean().item(),
        "token_bytes_std": token_bytes_nonzero.std().item(),
    }
])
