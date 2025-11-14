"""
SmolTalk: A high-quality conversational dataset for chat model training.

SmolTalk is a curated dataset of multi-turn conversations designed to teach models
general conversational abilities. It's part of HuggingFace's "smol" series, which
focuses on quality over quantity and is optimized for smaller models.

Dataset: https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk
Size: ~460K training conversations, ~24K test conversations

What makes SmolTalk useful:
- High-quality, diverse conversations covering many topics
- Multi-turn dialogues (not just single Q&A pairs)
- Natural conversational flow and context
- Filtered and curated to remove low-quality examples
- Size is manageable for smaller models (not overwhelming)

Conversation characteristics:
- Optional system messages at the start
- Alternating user/assistant messages
- Various lengths (2 to many turns)
- Diverse topics: general knowledge, advice, creative tasks, etc.

Why use SmolTalk:
- Teaches general chat capabilities
- Complements task-specific datasets (MMLU, GSM8K, etc.)
- Helps models be helpful, harmless, and honest in conversations
- Good for supervised fine-tuning (SFT) phase of training

This is a key dataset for making models conversational and useful in open-ended
dialogue, as opposed to just being good at specific benchmarks.
"""

from datasets import load_dataset
from tasks.common import Task

class SmolTalk(Task):
    """
    SmolTalk: High-quality conversational dataset for general chat abilities.

    Loads multi-turn conversations that teach models to be helpful, engaging,
    and natural in open-ended dialogue. Train split has ~460K conversations,
    test split has ~24K conversations.
    """

    def __init__(self, split, **kwargs):
        """
        Initialize the SmolTalk dataset.

        Args:
            split: Either "train" or "test"
            **kwargs: Additional arguments passed to parent Task

        The dataset is shuffled with a fixed seed for reproducibility.
        """
        super().__init__(**kwargs)
        assert split in ["train", "test"], "SmolTalk split must be train|test"
        # Load from HuggingFace and shuffle for variety during training
        self.ds = load_dataset("HuggingFaceTB/smol-smoltalk", split=split).shuffle(seed=42)
        self.length = len(self.ds)

    def num_examples(self):
        """Returns the total number of conversations in the dataset."""
        return self.length

    def get_example(self, index):
        """
        Retrieve a single conversation from the dataset.

        Args:
            index: Index into the dataset

        Returns:
            dict: Conversation with 'messages' field

        SmolTalk conversations have a specific structure:
        - Optional system message at the start (role="system")
        - Followed by alternating user/assistant messages
        - All content is simple strings (no structured parts)

        The function validates this structure to catch data issues early.
        """
        row = self.ds[index]
        messages = row["messages"]

        # ---------------------------------------------------------------------
        # Validation: Ensure conversation structure is correct
        # These asserts help catch data corruption or format changes early
        # TODO: Could be removed later once we're confident in data quality
        # ---------------------------------------------------------------------

        assert len(messages) >= 1, "Conversation must have at least 1 message"

        # Check if there's an optional system message at the start
        first_message = messages[0]
        if first_message["role"] == "system":
            # System message is optional but valid
            rest_messages = messages[1:]
        else:
            # No system message, all messages should be user/assistant
            rest_messages = messages

        # Must have at least one user-assistant exchange
        assert len(rest_messages) >= 2, \
            "SmolTalk conversation must have at least 2 non-system messages"

        # Validate that user and assistant alternate correctly
        for i, message in enumerate(rest_messages):
            # Should alternate: user, assistant, user, assistant, ...
            expected_role = "user" if i % 2 == 0 else "assistant"
            assert message["role"] == expected_role, \
                f"Message {i} has role {message['role']} but should be {expected_role}"

            # Content should be simple strings (not structured parts with tool calls)
            assert isinstance(message["content"], str), \
                f"Message {i} content must be a string, got {type(message['content'])}"

        # ---------------------------------------------------------------------
        # Return the conversation (including optional system message)
        # ---------------------------------------------------------------------
        conversation = {
            "messages": messages,
        }
        return conversation
