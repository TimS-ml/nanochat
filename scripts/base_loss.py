"""
Base Model Loss Evaluation and Sampling Script

This script loads a pretrained base model checkpoint and performs two tasks:
1. Evaluates loss (in bits per byte) on train and validation splits
2. Generates sample completions from the model on common prompts

The bits-per-byte (bpb) metric is preferable to standard loss because it's invariant
to the tokenizer's vocabulary size, making it easier to compare models with different
tokenizers.

This is useful for:
- Checking model quality after training
- Comparing different checkpoints
- Debugging model behavior through samples
- Measuring generalization (train vs val loss)

Usage examples:

Distributed evaluation (recommended):
    torchrun --standalone --nproc_per_node=8 -m scripts.base_loss

Single GPU:
    python -m scripts.base_loss

Evaluate specific checkpoint:
    python -m scripts.base_loss --model_tag=d20 --model_step=10000

Adjust evaluation size:
    python -m scripts.base_loss --split_tokens=10485760  # 10M tokens

The script will:
1. Load the model from base_checkpoints/
2. Evaluate loss on train/val splits (distributed across GPUs)
3. Generate sample completions (on master process only)
4. Print results to console and log to experiment report
"""
import os
from contextlib import nullcontext
import torch
from nanochat.checkpoint_manager import load_model
from nanochat.common import compute_init, print0, compute_cleanup, autodetect_device_type
from nanochat.dataloader import tokenizing_distributed_data_loader
from nanochat.tokenizer import get_token_bytes
from nanochat.loss_eval import evaluate_bpb
from nanochat.engine import Engine

# Configuration
device_batch_size = 32
split_tokens = 20*524288  # number of tokens to evaluate per split
model_tag = None # optional model tag for the output directory name
model_step = None # optional model step for the output directory name
device_type = "" # cuda|cpu|mps (empty => autodetect)
exec(open(os.path.join('nanochat', 'configurator.py')).read()) # overrides from command line or config file

# Load the base model and the tokenizer
device_type = autodetect_device_type() if device_type == "" else device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
model, tokenizer, meta = load_model("base", device, phase="eval", model_tag=model_tag, step=model_step)
sequence_len = meta["model_config"]["sequence_len"] # could be arbitrary really
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()

# Evaluate the loss on each split
tokens_per_step = device_batch_size * sequence_len * ddp_world_size
assert split_tokens % tokens_per_step == 0, "split_tokens must be divisible by tokens_per_step"
steps = split_tokens // tokens_per_step
token_bytes = get_token_bytes(device=device)
bpb_results = {}
for split_name in ["train", "val"]:
    loader = tokenizing_distributed_data_loader(device_batch_size, sequence_len, split_name, device=device)
    with autocast_ctx:
        bpb = evaluate_bpb(model, loader, steps, token_bytes)
    print0(f"{split_name} bpb: {bpb:.4f}")
    bpb_results[split_name] = bpb

# Master process also samples from the model
samples = []
if ddp_rank == 0:
    prompts = [
        "The capital of France is",
        "The chemical symbol of gold is",
        "If yesterday was Friday, then tomorrow will be",
        "The opposite of hot is",
        "The planets of the solar system are:",
        "My favorite color is",
        "If 5*x + 3 = 13, then x is",
    ]
    engine = Engine(model, tokenizer)
    for prompt in prompts:
        tokens = tokenizer(prompt, prepend="<|bos|>")
        with autocast_ctx:
            sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=16, temperature=0)
        sample_str = tokenizer.decode(sample[0])
        print0(sample_str)
        samples.append(sample_str)

# Log to report
from nanochat.report import get_report
get_report().log(section="Base model loss", data=[
    {
        "train bpb": bpb_results["train"],
        "val bpb": bpb_results["val"],
    },
    {f"sample {i}": sample for i, sample in enumerate(samples)},
])

# Cleanup
compute_cleanup()
