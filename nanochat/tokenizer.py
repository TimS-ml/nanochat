"""
BPE Tokenizer in the style of GPT-4.

Two implementations are available:
1) HuggingFace Tokenizer that can do both training and inference but is really confusing
2) Our own RustBPE Tokenizer for training and tiktoken for efficient inference
"""

import os
import copy
from functools import lru_cache

SPECIAL_TOKENS = [
    # every document begins with the Beginning of Sequence (BOS) token that delimits documents
    "<|bos|>",
    # tokens below are only used during finetuning to render Conversations into token ids
    "<|user_start|>", # user messages
    "<|user_end|>",
    "<|assistant_start|>", # assistant messages
    "<|assistant_end|>",
    "<|python_start|>", # assistant invokes python REPL tool
    "<|python_end|>",
    "<|output_start|>", # python REPL outputs back to assistant
    "<|output_end|>",
]

# NOTE: this split pattern deviates from GPT-4 in that we use \p{N}{1,2} instead of \p{N}{1,3}
# I did this because I didn't want to "waste" too many tokens on numbers for smaller vocab sizes.
# I haven't validated that this is actually a good idea, TODO.
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

# -----------------------------------------------------------------------------
# Generic GPT-4-style tokenizer based on HuggingFace Tokenizer
from tokenizers import Tokenizer as HFTokenizer
from tokenizers import pre_tokenizers, decoders, Regex
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

class HuggingFaceTokenizer:
    """
    Light wrapper around HuggingFace Tokenizer providing a unified interface.

    This class wraps the HuggingFace Tokenizers library to provide consistent
    methods for training, encoding, decoding, and managing BPE tokenizers.
    """

    def __init__(self, tokenizer):
        """
        Initialize the tokenizer wrapper.

        Args:
            tokenizer: An instance of HuggingFace Tokenizer
        """
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(cls, hf_path):
        """
        Load a tokenizer from HuggingFace's model hub.

        Args:
            hf_path: HuggingFace model identifier (e.g. "gpt2")

        Returns:
            HuggingFaceTokenizer instance
        """
        tokenizer = HFTokenizer.from_pretrained(hf_path)
        return cls(tokenizer)

    @classmethod
    def from_directory(cls, tokenizer_dir):
        """
        Load a tokenizer from a local directory.

        Args:
            tokenizer_dir: Path to directory containing tokenizer.json

        Returns:
            HuggingFaceTokenizer instance
        """
        tokenizer_path = os.path.join(tokenizer_dir, "tokenizer.json")
        tokenizer = HFTokenizer.from_file(tokenizer_path)
        return cls(tokenizer)

    @classmethod
    def train_from_iterator(cls, text_iterator, vocab_size):
        """
        Train a new BPE tokenizer from scratch using an iterator of text.

        This method configures and trains a GPT-4 style tokenizer with:
        - Byte-level BPE encoding (ensures all text can be represented)
        - GPT-4 regex splitting pattern
        - No normalization (preserves original text)
        - Special tokens for conversation formatting

        Args:
            text_iterator: Iterator yielding strings of text to train on
            vocab_size: Target vocabulary size (includes special tokens)

        Returns:
            HuggingFaceTokenizer instance with trained tokenizer
        """
        # Configure the HuggingFace Tokenizer with byte-level BPE
        # byte_fallback=True ensures every byte can be represented, preventing UNK tokens
        tokenizer = HFTokenizer(BPE(
            byte_fallback=True, # needed to handle any byte sequence
            unk_token=None,
            fuse_unk=False,
        ))
        # Normalizer: None - we don't want to modify the input text (no lowercasing, etc.)
        tokenizer.normalizer = None
        # Pre-tokenizer: GPT-4 style regex-based splitting
        # This regex splits text into chunks before applying BPE merges
        # NOTE: The pattern was changed from \p{N}{1,3} to \p{N}{1,2} because I suspect it is harmful to
        # very small models and smaller vocab sizes, because it is a little bit wasteful in the token space.
        # (but I haven't validated this! TODO)
        gpt4_split_regex = Regex(SPLIT_PATTERN) # huggingface demands that you wrap it in Regex!!
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            # First split by the regex pattern to isolate different text categories
            pre_tokenizers.Split(pattern=gpt4_split_regex, behavior="isolated", invert=False),
            # Then apply byte-level encoding to handle all Unicode properly
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
        ])
        # Decoder: ByteLevel (it pairs together with the ByteLevel pre-tokenizer)
        tokenizer.decoder = decoders.ByteLevel()
        # Post-processor: None - no additional processing after tokenization
        tokenizer.post_processor = None
        # Trainer: BPE configuration
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            show_progress=True,
            min_frequency=0, # no minimum frequency requirement for merges
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), # start with all possible bytes
            special_tokens=SPECIAL_TOKENS, # add conversation special tokens
        )
        # Kick off the training process
        tokenizer.train_from_iterator(text_iterator, trainer)
        return cls(tokenizer)

    def get_vocab_size(self):
        """Get the total vocabulary size including special tokens."""
        return self.tokenizer.get_vocab_size()

    def get_special_tokens(self):
        """
        Get list of all special tokens in the vocabulary.

        Returns:
            List of special token strings
        """
        special_tokens_map = self.tokenizer.get_added_tokens_decoder()
        special_tokens = [w.content for w in special_tokens_map.values()]
        return special_tokens

    def id_to_token(self, id):
        """
        Convert a token ID to its string representation.

        Args:
            id: Token ID (integer)

        Returns:
            String representation of the token
        """
        return self.tokenizer.id_to_token(id)

    def _encode_one(self, text, prepend=None, append=None):
        """
        Encode a single string to token IDs with optional special tokens.

        Args:
            text: String to encode
            prepend: Optional token or token ID to prepend (e.g. BOS token)
            append: Optional token or token ID to append (e.g. EOS token)

        Returns:
            List of token IDs
        """
        assert isinstance(text, str)
        ids = []
        # Add prepend token if specified
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.encode_special(prepend)
            ids.append(prepend_id)
        # Encode the main text without automatic special tokens
        ids.extend(self.tokenizer.encode(text, add_special_tokens=False).ids)
        # Add append token if specified
        if append is not None:
            append_id = append if isinstance(append, int) else self.encode_special(append)
            ids.append(append_id)
        return ids

    def encode_special(self, text):
        """
        Encode a single special token via exact string match.

        Args:
            text: Special token string (e.g. "<|bos|>")

        Returns:
            Token ID for the special token
        """
        return self.tokenizer.token_to_id(text)

    def get_bos_token_id(self):
        """
        Get the token ID for the beginning-of-sequence token.

        Returns:
            BOS token ID
        """
        bos = self.encode_special("<|bos|>")
        return bos

    def encode(self, text, *args, **kwargs):
        """
        Encode text to token IDs. Handles both single strings and lists of strings.

        Args:
            text: String or list of strings to encode
            *args, **kwargs: Additional arguments passed to _encode_one

        Returns:
            List of token IDs (if text is string) or list of lists (if text is list)
        """
        if isinstance(text, str):
            return self._encode_one(text, *args, **kwargs)
        elif isinstance(text, list):
            return [self._encode_one(t, *args, **kwargs) for t in text]
        else:
            raise ValueError(f"Invalid input type: {type(text)}")

    def __call__(self, *args, **kwargs):
        """Allow calling the tokenizer instance directly like a function."""
        return self.encode(*args, **kwargs)

    def decode(self, ids):
        """
        Decode token IDs back to text.

        Args:
            ids: List of token IDs

        Returns:
            Decoded string (includes special tokens)
        """
        return self.tokenizer.decode(ids, skip_special_tokens=False)

    def save(self, tokenizer_dir):
        """
        Save the tokenizer to disk.

        Args:
            tokenizer_dir: Directory path to save tokenizer.json
        """
        os.makedirs(tokenizer_dir, exist_ok=True)
        tokenizer_path = os.path.join(tokenizer_dir, "tokenizer.json")
        self.tokenizer.save(tokenizer_path)
        print(f"Saved tokenizer to {tokenizer_path}")

# -----------------------------------------------------------------------------
# Tokenizer based on rustbpe + tiktoken combo
import pickle
import rustbpe
import tiktoken

class RustBPETokenizer:
    """
    Tokenizer using rustbpe for training and tiktoken for efficient inference.

    This implementation provides faster training with rustbpe and faster encoding
    with tiktoken compared to the HuggingFace implementation. It's the recommended
    tokenizer for production use in nanochat.
    """

    def __init__(self, enc, bos_token):
        """
        Initialize the tokenizer with a tiktoken encoding.

        Args:
            enc: tiktoken.Encoding instance
            bos_token: String representation of the BOS token (e.g. "<|bos|>")
        """
        self.enc = enc
        self.bos_token_id = self.encode_special(bos_token)

    @classmethod
    def train_from_iterator(cls, text_iterator, vocab_size):
        """
        Train a new BPE tokenizer from scratch using rustbpe, then convert to tiktoken.

        The training process:
        1. Train BPE merges using the fast Rust implementation
        2. Reserve space for special tokens at the end of the vocabulary
        3. Convert the trained tokenizer to tiktoken format for efficient inference

        Args:
            text_iterator: Iterator yielding strings of text to train on
            vocab_size: Target vocabulary size (includes special tokens)

        Returns:
            RustBPETokenizer instance ready for encoding/decoding
        """
        # 1) Train using rustbpe (fast Rust-based BPE training)
        tokenizer = rustbpe.Tokenizer()
        # Reserve space for special tokens at the end of the vocabulary
        # The special tokens are inserted later, we don't train them here
        vocab_size_no_special = vocab_size - len(SPECIAL_TOKENS)
        assert vocab_size_no_special >= 256, f"vocab_size_no_special must be at least 256, got {vocab_size_no_special}"
        tokenizer.train_from_iterator(text_iterator, vocab_size_no_special, pattern=SPLIT_PATTERN)
        # 2) Construct the associated tiktoken encoding for efficient inference
        pattern = tokenizer.get_pattern()
        mergeable_ranks_list = tokenizer.get_mergeable_ranks()
        # Convert to the format expected by tiktoken
        mergeable_ranks = {bytes(k): v for k, v in mergeable_ranks_list}
        # Add special tokens after all regular tokens
        tokens_offset = len(mergeable_ranks)
        special_tokens = {name: tokens_offset + i for i, name in enumerate(SPECIAL_TOKENS)}
        # Create tiktoken encoding with learned BPE merges and special tokens
        enc = tiktoken.Encoding(
            name="rustbpe",
            pat_str=pattern,
            mergeable_ranks=mergeable_ranks, # dict[bytes, int] (token bytes -> merge priority rank)
            special_tokens=special_tokens, # dict[str, int] (special token name -> token id)
        )
        return cls(enc, "<|bos|>")

    @classmethod
    def from_directory(cls, tokenizer_dir):
        """
        Load a tokenizer from a local directory.

        Args:
            tokenizer_dir: Path to directory containing tokenizer.pkl

        Returns:
            RustBPETokenizer instance
        """
        pickle_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
        with open(pickle_path, "rb") as f:
            enc = pickle.load(f)
        return cls(enc, "<|bos|>")

    @classmethod
    def from_pretrained(cls, tiktoken_name):
        """
        Load a pretrained tiktoken tokenizer (e.g. for GPT-2, GPT-3, GPT-4).

        Args:
            tiktoken_name: Name of tiktoken encoding (e.g. "gpt2", "cl100k_base")

        Returns:
            RustBPETokenizer instance

        Note:
            OpenAI's tiktoken uses "<|endoftext|>" as the document delimiter token,
            which is confusing because this token is typically PREPENDED to documents
            to signal the start of a new sequence. In nanochat we prefer to call it
            "<|bos|>" (beginning of sequence), but for compatibility with OpenAI
            tokenizers we use their naming convention here.
        """
        # Reference: https://github.com/openai/tiktoken/blob/eedc8563/tiktoken_ext/openai_public.py
        enc = tiktoken.get_encoding(tiktoken_name)
        # tiktoken calls the special document delimiter token "<|endoftext|>"
        # yes this is confusing because this token is almost always PREPENDED to the beginning of the document
        # it most often is used to signal the start of a new sequence to the LLM during inference etc.
        # so in nanoChat we always use "<|bos|>" short for "beginning of sequence", but historically it is often called "<|endoftext|>".
        return cls(enc, "<|endoftext|>")

    def get_vocab_size(self):
        """Get the total vocabulary size including special tokens."""
        return self.enc.n_vocab

    def get_special_tokens(self):
        """
        Get set of all special tokens in the vocabulary.

        Returns:
            Set of special token strings
        """
        return self.enc.special_tokens_set

    def id_to_token(self, id):
        """
        Convert a token ID to its string representation.

        Args:
            id: Token ID (integer)

        Returns:
            String representation of the token (decoded bytes)
        """
        return self.enc.decode([id])

    @lru_cache(maxsize=32)
    def encode_special(self, text):
        """
        Encode a single special token via exact string match.

        Args:
            text: Special token string (e.g. "<|bos|>")

        Returns:
            Token ID for the special token

        Note:
            Results are cached for efficiency since special tokens are used frequently.
        """
        return self.enc.encode_single_token(text)

    def get_bos_token_id(self):
        """
        Get the token ID for the beginning-of-sequence token.

        Returns:
            BOS token ID
        """
        return self.bos_token_id

    def encode(self, text, prepend=None, append=None, num_threads=8):
        """
        Encode text to token IDs using tiktoken's fast encoding.

        Args:
            text: String or list of strings to encode
            prepend: Optional token or token ID to prepend (e.g. BOS token)
            append: Optional token or token ID to append (e.g. EOS token)
            num_threads: Number of threads for batch encoding (default: 8)

        Returns:
            List of token IDs (if text is string) or list of lists (if text is list)

        Note:
            tiktoken's encode_ordinary_batch provides significant speedup for
            encoding multiple strings compared to encoding them one by one.
        """
        # Resolve prepend/append to token IDs if they're strings
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.encode_special(prepend)
        if append is not None:
            append_id = append if isinstance(append, int) else self.encode_special(append)

        if isinstance(text, str):
            # Single string: use ordinary encoding (no special token processing)
            ids = self.enc.encode_ordinary(text)
            if prepend is not None:
                ids.insert(0, prepend_id) # TODO: slightly inefficient here? :( hmm
            if append is not None:
                ids.append(append_id)
        elif isinstance(text, list):
            # Batch encoding: tiktoken can process multiple strings in parallel
            ids = self.enc.encode_ordinary_batch(text, num_threads=num_threads)
            if prepend is not None:
                for ids_row in ids:
                    ids_row.insert(0, prepend_id) # TODO: same
            if append is not None:
                for ids_row in ids:
                    ids_row.append(append_id)
        else:
            raise ValueError(f"Invalid input type: {type(text)}")

        return ids

    def __call__(self, *args, **kwargs):
        """Allow calling the tokenizer instance directly like a function."""
        return self.encode(*args, **kwargs)

    def decode(self, ids):
        """
        Decode token IDs back to text.

        Args:
            ids: List of token IDs

        Returns:
            Decoded string (includes special tokens)
        """
        return self.enc.decode(ids)

    def save(self, tokenizer_dir):
        """
        Save the tokenizer encoding to disk as a pickle file.

        Args:
            tokenizer_dir: Directory path to save tokenizer.pkl
        """
        os.makedirs(tokenizer_dir, exist_ok=True)
        pickle_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
        with open(pickle_path, "wb") as f:
            pickle.dump(self.enc, f)
        print(f"Saved tokenizer encoding to {pickle_path}")

    def render_conversation(self, conversation, max_tokens=2048):
        """
        Tokenize a chat conversation into token IDs with training masks.

        This method converts a structured conversation (with user/assistant messages)
        into a sequence of token IDs suitable for training. The mask indicates which
        tokens should be used for loss calculation during training.

        Args:
            conversation: Dict with 'messages' key containing list of message dicts
                         Each message has 'role' (user/assistant/system) and 'content'
            max_tokens: Maximum sequence length to prevent OOMs (default: 2048)

        Returns:
            Tuple of (ids, mask):
            - ids: list[int] of token IDs representing the conversation
            - mask: list[int] of same length, where 1 = train on this token, 0 = don't train

        Training Mask Strategy:
            - mask=0: User messages, special tokens, system messages, tool outputs
            - mask=1: Assistant messages (what we want the model to learn to generate)

        Conversation Format:
            <|bos|><|user_start|>user text<|user_end|><|assistant_start|>assistant text<|assistant_end|>
        """
        # ids and masks that we will return, plus a helper function to build them up
        ids, mask = [], []
        def add_tokens(token_ids, mask_val):
            """Helper to add tokens and their corresponding mask values."""
            if isinstance(token_ids, int):
                token_ids = [token_ids]
            ids.extend(token_ids)
            mask.extend([mask_val] * len(token_ids))

        # Handle system messages by merging them with the first user message
        # System messages provide context but aren't part of the conversation flow
        if conversation["messages"][0]["role"] == "system":
            # Deep copy to avoid mutating the original conversation
            conversation = copy.deepcopy(conversation)
            messages = conversation["messages"]
            assert messages[1]["role"] == "user", "System message must be followed by a user message"
            # Merge system message content into the user message
            messages[1]["content"] = messages[0]["content"] + "\n\n" + messages[1]["content"]
            messages = messages[1:]  # Skip the system message
        else:
            messages = conversation["messages"]
        assert len(messages) >= 1, f"Conversation has less than 1 message: {messages}"

        # Fetch all the special tokens we need for conversation formatting
        bos = self.get_bos_token_id()
        user_start, user_end = self.encode_special("<|user_start|>"), self.encode_special("<|user_end|>")
        assistant_start, assistant_end = self.encode_special("<|assistant_start|>"), self.encode_special("<|assistant_end|>")
        python_start, python_end = self.encode_special("<|python_start|>"), self.encode_special("<|python_end|>")
        output_start, output_end = self.encode_special("<|output_start|>"), self.encode_special("<|output_end|>")

        # Begin with BOS token (mask=0, we don't train on special tokens)
        add_tokens(bos, 0)

        # Process each message in the conversation
        for i, message in enumerate(messages):
            # Enforce alternating user/assistant pattern to prevent training errors
            # Even indices (0, 2, 4...) should be user, odd indices (1, 3, 5...) should be assistant
            must_be_from = "user" if i % 2 == 0 else "assistant"
            assert message["role"] == must_be_from, f"Message {i} is from {message['role']} but should be from {must_be_from}"

            # Content can be either a simple string or a list of parts (for tool calls)
            content = message["content"]

            if message["role"] == "user":
                # User messages: wrapped in user_start/user_end tokens, mask=0 (don't train)
                assert isinstance(content, str), "User messages are simply expected to be strings"
                value_ids = self.encode(content)
                add_tokens(user_start, 0)
                add_tokens(value_ids, 0)  # Don't train on user input
                add_tokens(user_end, 0)
            elif message["role"] == "assistant":
                # Assistant messages: wrapped in assistant_start/assistant_end, mask=1 (train)
                add_tokens(assistant_start, 0)
                if isinstance(content, str):
                    # Simple text response from assistant
                    value_ids = self.encode(content)
                    add_tokens(value_ids, 1)  # Train on assistant responses
                elif isinstance(content, list):
                    # Complex response with multiple parts (text, tool calls, outputs)
                    for part in content:
                        value_ids = self.encode(part["text"])
                        if part["type"] == "text":
                            # Regular text part
                            add_tokens(value_ids, 1)
                        elif part["type"] == "python":
                            # Python tool call (train the model to generate this)
                            add_tokens(python_start, 1)
                            add_tokens(value_ids, 1)
                            add_tokens(python_end, 1)
                        elif part["type"] == "python_output":
                            # Python execution output (comes from the REPL, not the model)
                            # mask=0 because the model doesn't generate this, Python does
                            add_tokens(output_start, 0)
                            add_tokens(value_ids, 0)
                            add_tokens(output_end, 0)
                        else:
                            raise ValueError(f"Unknown part type: {part['type']}")
                else:
                    raise ValueError(f"Unknown content type: {type(content)}")
                add_tokens(assistant_end, 1)

        # Truncate to max_tokens to prevent OOMs during training
        ids = ids[:max_tokens]
        mask = mask[:max_tokens]
        return ids, mask

    def visualize_tokenization(self, ids, mask, with_token_id=False):
        """
        Visualize tokenization with color-coded tokens for debugging.

        This helper function is useful when debugging conversation rendering.
        It shows which tokens will be trained on (green) vs ignored (red).

        Args:
            ids: List of token IDs
            mask: List of mask values (0 or 1)
            with_token_id: If True, show token IDs in gray (default: False)

        Returns:
            String with ANSI color codes for terminal display
            - Green tokens: mask=1 (trained on)
            - Red tokens: mask=0 (not trained on)
        """
        RED = '\033[91m'
        GREEN = '\033[92m'
        RESET = '\033[0m'
        GRAY = '\033[90m'
        tokens = []
        for i, (token_id, mask_val) in enumerate(zip(ids, mask)):
            token_str = self.decode([token_id])
            color = GREEN if mask_val == 1 else RED
            tokens.append(f"{color}{token_str}{RESET}")
            if with_token_id:
                tokens.append(f"{GRAY}({token_id}){RESET}")
        return '|'.join(tokens)

    def render_for_completion(self, conversation):
        """
        Render a conversation for RL completion (priming the assistant to respond).

        Used during Reinforcement Learning where we want the model to generate
        a completion for a conversation. This removes the last assistant message
        and adds the assistant_start token to prime the model for generation.

        Args:
            conversation: Dict with 'messages' key, where last message is from assistant

        Returns:
            List of token IDs ending with <|assistant_start|>, ready for completion

        Note:
            Unlike render_conversation, this doesn't return a mask since we're
            generating completions, not training with supervised learning.
        """
        # Remove the last assistant message (we'll ask the model to generate it)
        conversation = copy.deepcopy(conversation)  # avoid mutating the original
        messages = conversation["messages"]
        assert messages[-1]["role"] == "assistant", "Last message must be from the Assistant"
        messages.pop()  # remove the last message inplace

        # Tokenize the conversation up to (but not including) the assistant's response
        ids, mask = self.render_conversation(conversation)

        # Prime the model to generate an assistant response
        assistant_start = self.encode_special("<|assistant_start|>")
        ids.append(assistant_start)
        return ids

# -----------------------------------------------------------------------------
# nanochat-specific convenience functions

def get_tokenizer():
    """
    Load the nanochat tokenizer from the configured base directory.

    Returns:
        RustBPETokenizer instance loaded from disk

    Note:
        Reads from $NANOCHAT_BASE_DIR/tokenizer/ directory.
        The tokenizer must have been previously trained and saved.
    """
    from nanochat.common import get_base_dir
    base_dir = get_base_dir()
    tokenizer_dir = os.path.join(base_dir, "tokenizer")
    # return HuggingFaceTokenizer.from_directory(tokenizer_dir)
    return RustBPETokenizer.from_directory(tokenizer_dir)

def get_token_bytes(device="cpu"):
    """
    Load the token_bytes tensor for bits-per-byte calculations.

    The token_bytes tensor maps each token ID to its byte length, used for
    calculating vocab-size-independent loss metrics (bits per byte).

    Args:
        device: Device to load the tensor onto (default: "cpu")

    Returns:
        torch.Tensor of shape (vocab_size,) where each element is the byte
        length of that token, or 0 for special tokens that should be ignored

    Note:
        This file is generated by tok_train.py during tokenizer training.
    """
    import torch
    from nanochat.common import get_base_dir
    base_dir = get_base_dir()
    tokenizer_dir = os.path.join(base_dir, "tokenizer")
    token_bytes_path = os.path.join(tokenizer_dir, "token_bytes.pt")
    assert os.path.exists(token_bytes_path), f"Token bytes not found at {token_bytes_path}? It gets written by tok_train.py"
    with open(token_bytes_path, "rb") as f:
        token_bytes = torch.load(f, map_location=device)
    return token_bytes
