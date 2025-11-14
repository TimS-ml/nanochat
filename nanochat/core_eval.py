"""
Functions for evaluating the CORE metric, as described in the DCLM paper.
https://arxiv.org/abs/2406.11794

TODOs:
- All tasks ~match except for squad. We get 31% reference is 37%. Figure out why.
"""
import random

from jinja2 import Template
import torch
import torch.distributed as dist

# -----------------------------------------------------------------------------
# Prompt rendering utilities

def render_prompts_mc(item, continuation_delimiter, fewshot_examples=None):
    """
    Render prompts for a multiple choice question.

    Creates one prompt per answer choice, with optional few-shot examples.
    Each prompt contains the question followed by one of the answer choices.

    Args:
        item: Dict with 'query' and 'choices' keys
        continuation_delimiter: String separating question from answer (e.g. " ")
        fewshot_examples: Optional list of example items to prepend

    Returns:
        List of prompts, one for each choice in item['choices']

    Example:
        item = {'query': 'What is 2+2?', 'choices': ['3', '4', '5'], 'gold': 1}
        prompts = render_prompts_mc(item, ' ')
        # Returns: ['What is 2+2? 3', 'What is 2+2? 4', 'What is 2+2? 5']
    """
    template_str = """
{%- for example in fewshot_examples -%}
{{ example.query }}{{ continuation_delimiter }}{{ example.choices[example.gold] }}

{% endfor -%}
{{ item.query }}{{ continuation_delimiter }}{{ choice }}""".strip()
    template = Template(template_str)
    fewshot_examples = fewshot_examples or []
    context = {
        'fewshot_examples': fewshot_examples,
        'continuation_delimiter': continuation_delimiter,
        'item': item
    }
    # Generate one prompt per answer choice
    prompts = [template.render(choice=choice, **context) for choice in item['choices']]
    return prompts


def render_prompts_schema(item, continuation_delimiter, fewshot_examples=None):
    """
    Render prompts for a schema/cloze-style question.

    Creates one prompt per context option. Unlike multiple choice where the
    choices vary, here the continuation is fixed but the context varies.

    Args:
        item: Dict with 'context_options' and 'continuation' keys
        continuation_delimiter: String separating context from continuation
        fewshot_examples: Optional list of example items to prepend

    Returns:
        List of prompts, one for each context option

    Example:
        item = {
            'context_options': ['The cat sat on the', 'The dog sat on the'],
            'continuation': 'mat',
            'gold': 0
        }
        # Returns prompts with each context followed by 'mat'
    """
    template_str = """
{%- for example in fewshot_examples -%}
{{ example.context_options[example.gold] }}{{ continuation_delimiter }}{{ example.continuation }}

{% endfor -%}
{{ context }}{{ continuation_delimiter }}{{ item.continuation }}""".strip()
    template = Template(template_str)
    fewshot_examples = fewshot_examples or []
    context = {
        'fewshot_examples': fewshot_examples,
        'continuation_delimiter': continuation_delimiter,
        'item': item
    }
    # Generate one prompt per context option
    prompts = [template.render(context=context_option, **context)
               for context_option in item['context_options']]
    return prompts


def render_prompts_lm(item, continuation_delimiter, fewshot_examples=None):
    """
    Render prompts for a language modeling task.

    Returns two prompts: one without the continuation (for measuring loss)
    and one with it (for extracting the continuation tokens).

    Args:
        item: Dict with 'context' and 'continuation' keys
        continuation_delimiter: String separating context from continuation
        fewshot_examples: Optional list of example items to prepend

    Returns:
        List of two prompts: [prompt_without_continuation, prompt_with_continuation]

    Note:
        We trim whitespace to ensure clean tokenization boundaries. Without
        this, trailing whitespace in the context can get absorbed into the
        first continuation token, making it impossible to identify where
        the continuation starts in token space.
    """
    template_str = """
{%- for example in fewshot_examples -%}
{{ example.context | trim }}{{ continuation_delimiter }}{{ example.continuation }}

{% endfor -%}
{{ item.context | trim }}{{ continuation_delimiter }}{% if include_continuation %}{{ item.continuation }}{% endif %}""".strip()
    template = Template(template_str)
    fewshot_examples = fewshot_examples or []
    context = {
        'fewshot_examples': fewshot_examples,
        'continuation_delimiter': continuation_delimiter,
        'item': item
    }
    # Return two prompts: without and with the continuation
    prompt_without = template.render(include_continuation=False, **context)
    prompt_with = template.render(include_continuation=True, **context)
    # Due to the way the data seems to be stored, I think I need to strip in the case of LM here.
    # Otherwise we may get trailing whitespaces in prompt_without (which get absorbed into the next
    # token in prompt_with), meaning we don't get a nice and clean prefix in the token space
    # to detect the final continuation. Tokenizers...
    prompt_without = prompt_without.strip()
    return [prompt_without, prompt_with]


def find_common_length(token_sequences, direction='left'):
    """
    Find the length of common prefix or suffix across multiple token sequences.

    This is used to identify where answer choices diverge in multiple choice
    questions, or where contexts converge in schema questions.

    Args:
        token_sequences: List of token ID lists
        direction: 'left' for common prefix, 'right' for common suffix

    Returns:
        Integer length of the common portion

    Example:
        >>> seqs = [[1, 2, 3, 4], [1, 2, 5, 6]]
        >>> find_common_length(seqs, 'left')
        2  # First 2 tokens are common
    """
    min_len = min(len(seq) for seq in token_sequences)
    indices = {
        'left': range(min_len),  # Scan from left to right
        'right': range(-1, -min_len-1, -1)  # Scan from right to left
    }[direction]
    # Find the first position where the token sequences differ
    for i, idx in enumerate(indices):
        token = token_sequences[0][idx]
        if not all(seq[idx] == token for seq in token_sequences):
            return i  # Found where they diverge
    return min_len  # All common


def stack_sequences(tokens, pad_token_id):
    """
    Stack variable-length token sequences into a padded batch tensor.

    Args:
        tokens: List of token ID lists (variable length)
        pad_token_id: Token ID to use for padding

    Returns:
        torch.Tensor of shape (batch_size, max_seq_len) with right-side padding
    """
    bsz, seq_len = len(tokens), max(len(x) for x in tokens)
    # Initialize with padding tokens
    input_ids = torch.full((bsz, seq_len), pad_token_id, dtype=torch.long)
    # Fill in actual token sequences (left-aligned)
    for i, x in enumerate(tokens):
        input_ids[i, :len(x)] = torch.tensor(x, dtype=torch.long)
    return input_ids


def batch_sequences_mc(tokenizer, prompts):
    """
    Tokenize and batch multiple choice prompts.

    In multiple choice, all prompts share a common question prefix but have
    different answer choices appended. We identify where the answers start
    so we can measure the loss only on the answer tokens.

    Args:
        tokenizer: Tokenizer instance
        prompts: List of prompt strings (one per answer choice)

    Returns:
        Tuple of (tokens, start_indices, end_indices):
        - tokens: List of token ID lists
        - start_indices: Where each answer starts (same for all)
        - end_indices: Where each sequence ends (may vary by answer length)
    """
    # Tokenize all prompts with BOS token prepended
    tokens = tokenizer(prompts, prepend=tokenizer.get_bos_token_id())
    # Find where the common prefix ends (where answers diverge)
    answer_start_idx = find_common_length(tokens, direction='left')
    start_indices = [answer_start_idx] * len(prompts)
    end_indices = [len(x) for x in tokens]
    return tokens, start_indices, end_indices


def batch_sequences_schema(tokenizer, prompts):
    """
    Tokenize and batch schema/cloze prompts.

    In schema tasks, contexts vary but the continuation is the same. We identify
    where the continuation starts so we can measure loss only on those tokens.

    Args:
        tokenizer: Tokenizer instance
        prompts: List of prompt strings (one per context option)

    Returns:
        Tuple of (tokens, start_indices, end_indices):
        - tokens: List of token ID lists
        - start_indices: Where each continuation starts (varies by context length)
        - end_indices: Where each sequence ends (same for all)
    """
    # Tokenize all prompts with BOS token prepended
    tokens = tokenizer(prompts, prepend=tokenizer.get_bos_token_id())
    # Find the length of the common suffix (the continuation)
    suffix_length = find_common_length(tokens, direction='right')
    end_indices = [len(x) for x in tokens]
    # Continuation starts suffix_length tokens from the end
    start_indices = [ei - suffix_length for ei in end_indices]
    return tokens, start_indices, end_indices


def batch_sequences_lm(tokenizer, prompts):
    """
    Tokenize and batch language modeling prompts.

    For LM tasks, we have two prompts: context only, and context+continuation.
    We verify they form a proper prefix relationship and identify where the
    continuation starts.

    Args:
        tokenizer: Tokenizer instance
        prompts: List of 2 strings: [context_only, context_with_continuation]

    Returns:
        Tuple of (tokens, start_indices, end_indices):
        - tokens: List containing only the full prompt tokens
        - start_indices: Where continuation starts (length of context)
        - end_indices: Where sequence ends (length of full prompt)
    """
    # Tokenize both prompts
    tokens = tokenizer(prompts, prepend=tokenizer.get_bos_token_id())
    tokens_without, tokens_with = tokens
    start_idx, end_idx = len(tokens_without), len(tokens_with)
    # Sanity checks to ensure proper prefix relationship
    assert start_idx < end_idx, "prompt without is supposed to be a prefix of prompt with"
    assert tokens_without == tokens_with[:start_idx], "prompt without is supposed to be a prefix of prompt with"
    # We only need the full prompt for LM task (batch size of 1)
    return [tokens_with], [start_idx], [end_idx]


@torch.no_grad()
def forward_model(model, input_ids):
    """
    Forward pass through the model to get per-token losses and predictions.

    Args:
        model: Language model with forward pass returning logits
        input_ids: Tensor of shape (batch_size, seq_len) containing token IDs

    Returns:
        Tuple of (losses, predictions):
        - losses: Tensor (batch_size, seq_len) of per-token cross entropy losses
                  Last column is NaN since there's no next token to predict
        - predictions: Tensor (batch_size, seq_len) of argmax predictions

    Note:
        Autoregressive models predict the next token, so:
        - losses[i, t] = loss for predicting input_ids[i, t+1]
        - predictions[i, t] = predicted token ID for position t+1
    """
    batch_size, seq_len = input_ids.size()
    # Forward pass to get logits (batch_size, seq_len, vocab_size)
    outputs = model(input_ids)
    # Shift tokens left by one to get autoregressive targets
    # target_ids[i, t] = input_ids[i, t+1] (what we're trying to predict)
    target_ids = torch.roll(input_ids, shifts=-1, dims=1)
    # Calculate cross entropy loss at all positions
    # Flatten to (batch_size * seq_len, vocab_size) for cross_entropy
    losses = torch.nn.functional.cross_entropy(
        outputs.view(batch_size * seq_len, -1),
        target_ids.view(batch_size * seq_len),
        reduction='none'  # Keep per-token losses
    ).view(batch_size, seq_len)
    # Set the last column to NaN because there's no next token to predict there
    losses[:, -1] = float('nan')
    # Get the argmax predictions at each position
    predictions = outputs.argmax(dim=-1)
    return losses, predictions


@torch.no_grad()
def evaluate_example(idx, model, tokenizer, data, device, task_meta):
    """
    Evaluate a single test example with the model.

    This function handles all task types (multiple choice, schema, language modeling)
    and supports few-shot prompting. It measures whether the model gets the
    example correct according to the task type's scoring method.

    Args:
        idx: Index of the example to evaluate
        model: Language model to evaluate
        tokenizer: Tokenizer for encoding prompts
        data: Full dataset (list of examples)
        device: Device to run evaluation on
        task_meta: Dict with task configuration:
            - task_type: 'multiple_choice', 'schema', or 'language_modeling'
            - num_fewshot: Number of few-shot examples to include
            - continuation_delimiter: String separating context from continuation

    Returns:
        bool: True if model answered correctly, False otherwise

    Scoring methods by task type:
        - multiple_choice/schema: Correct if gold option has lowest average loss
        - language_modeling: Correct if all continuation tokens predicted exactly
    """
    item = data[idx]
    task_type = task_meta['task_type']
    num_fewshot = task_meta['num_fewshot']
    continuation_delimiter = task_meta['continuation_delimiter']

    # Sample few-shot examples (excluding current item to avoid leakage)
    fewshot_examples = []
    if num_fewshot > 0:
        # Use deterministic random seed based on index for reproducibility
        rng = random.Random(1234 + idx)
        available_indices = [i for i in range(len(data)) if i != idx]
        fewshot_indices = rng.sample(available_indices, num_fewshot)
        fewshot_examples = [data[i] for i in fewshot_indices]

    # Render prompts and prepare batches based on task type
    if task_type == 'multiple_choice':
        prompts = render_prompts_mc(item, continuation_delimiter, fewshot_examples)
        tokens, start_idxs, end_idxs = batch_sequences_mc(tokenizer, prompts)
    elif task_type == 'schema':
        prompts = render_prompts_schema(item, continuation_delimiter, fewshot_examples)
        tokens, start_idxs, end_idxs = batch_sequences_schema(tokenizer, prompts)
    elif task_type == 'language_modeling':
        prompts = render_prompts_lm(item, continuation_delimiter, fewshot_examples)
        tokens, start_idxs, end_idxs = batch_sequences_lm(tokenizer, prompts)
    else:
        raise ValueError(f"Unsupported task type: {task_type}")

    # Handle models with maximum sequence length constraints (e.g. GPT-2)
    # If sequences are too long, truncate from the left (keeping the answer/continuation)
    if hasattr(model, 'max_seq_len') and model.max_seq_len is not None:
        max_tokens = model.max_seq_len
        new_tokens, new_start_idxs, new_end_idxs = [], [], []
        for t, s, e in zip(tokens, start_idxs, end_idxs):
            if len(t) > max_tokens:
                # Sequence too long: take only the last max_tokens tokens
                num_to_crop = len(t) - max_tokens
                new_tokens.append(t[-max_tokens:])  # Keep the end (has the answer)
                new_start_idxs.append(s - num_to_crop)  # Adjust indices
                new_end_idxs.append(e - num_to_crop)
                # Sanity checks: start/end should still be in valid range
                assert s - num_to_crop >= 0, "this should never happen right?"
                assert e - num_to_crop >= 0, "this should never happen right?"
            else:
                # Sequence fits: keep unchanged
                new_tokens.append(t)
                new_start_idxs.append(s)
                new_end_idxs.append(e)
        tokens, start_idxs, end_idxs = new_tokens, new_start_idxs, new_end_idxs

    # Stack all sequences into a padded batch tensor
    pad_token_id = tokenizer.get_bos_token_id()  # BOS token works fine as padding
    input_ids = stack_sequences(tokens, pad_token_id)
    input_ids = input_ids.to(device)

    # Forward pass: get per-token losses and predictions
    losses, predictions = forward_model(model, input_ids)

    # Determine if the model got this example correct (task-specific scoring)
    if task_type == 'language_modeling':
        # LM task: all continuation tokens must be predicted exactly
        si = start_idxs[0]
        ei = end_idxs[0]
        # predictions[t] predicts input_ids[t+1], so we need to offset by 1
        predicted_tokens = predictions[0, si-1:ei-1]
        actual_tokens = input_ids[0, si:ei]
        is_correct = torch.all(predicted_tokens == actual_tokens).item()
    elif task_type in ['multiple_choice', 'schema']:
        # MC/schema: correct answer is the one with lowest average loss
        # Calculate mean loss for each option's answer/continuation tokens
        mean_losses = [losses[i, si-1:ei-1].mean().item()
                        for i, (si, ei) in enumerate(zip(start_idxs, end_idxs))]
        pred_idx = mean_losses.index(min(mean_losses))  # Find lowest loss
        is_correct = pred_idx == item['gold']  # Compare to correct answer
    else:
        raise ValueError(f"Unsupported task type: {task_type}")

    return is_correct


def evaluate_task(model, tokenizer, data, device, task_meta):
    """
    Evaluate a model on an entire task dataset.

    This function evaluates all examples in the dataset and computes the
    overall accuracy. It supports distributed evaluation across multiple
    GPUs/processes for faster evaluation on large datasets.

    Args:
        model: Language model to evaluate
        tokenizer: Tokenizer for encoding prompts
        data: List of examples to evaluate
        device: Device to run evaluation on
        task_meta: Dict with task configuration (see evaluate_example)

    Returns:
        float: Accuracy (fraction of correct answers) in range [0, 1]

    Distributed Evaluation:
        If running with multiple processes (e.g. via torchrun):
        - Each process evaluates a subset of examples (strided by rank)
        - Results are aggregated across all processes via all_reduce
        - All processes end up with the same final accuracy value
    """
    # Get distributed training info (if applicable)
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    # Tensor to track correctness for each example
    correct = torch.zeros(len(data), dtype=torch.float32, device=device)

    # Each rank evaluates a strided subset of examples
    # E.g. with 4 ranks: rank 0 does [0,4,8,...], rank 1 does [1,5,9,...], etc.
    for idx in range(rank, len(data), world_size):
        is_correct = evaluate_example(idx, model, tokenizer, data, device, task_meta)
        correct[idx] = float(is_correct)

    # Synchronize results across all processes if running distributed
    if world_size > 1:
        dist.barrier()  # Ensure all ranks finish before aggregation
        # Sum the correct tensor across all ranks (each rank filled different indices)
        dist.all_reduce(correct, op=dist.ReduceOp.SUM)

    # Compute mean accuracy (same value on all ranks after all_reduce)
    mean_correct = correct.mean().item()
    return mean_correct
