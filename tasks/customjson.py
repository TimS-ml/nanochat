"""
CustomJSON: Load custom training data from JSONL files.

This task allows you to train models on your own conversational data by providing
a JSONL (JSON Lines) file where each line is a conversation. This is useful for:
- Domain-specific fine-tuning
- Custom instruction-following data
- Proprietary datasets
- Experimental conversation formats

File format:
- Extension: .jsonl (one JSON object per line)
- Each line: A JSON array of message objects
- Message format: {"role": "user"|"assistant", "content": "..."}
- Roles must alternate: user, assistant, user, assistant, ...

Example JSONL file content:
    [{"role":"user","content":"What is 2+2?"},{"role":"assistant","content":"4"}]
    [{"role":"user","content":"Hello!"},{"role":"assistant","content":"Hi there!"}]

Validation:
- Checks that each conversation has at least 2 messages
- Ensures roles alternate correctly (user/assistant/user/assistant)
- Validates message structure (role and content fields present)
- Ensures content is a string (not structured parts)

Use cases:
- Identity conversations (teaching models about themselves)
- Domain knowledge (e.g., medical, legal, technical conversations)
- Style transfer (teaching specific response patterns)
- Preference data for RLHF
"""

import os
import json
from tasks.common import Task

class CustomJSON(Task):
    """
    Load custom conversations from a JSONL (JSON Lines) file.

    This task enables training on custom conversational data by loading it from
    a simple JSONL file format. Each line is a complete conversation with alternating
    user/assistant messages.

    File format requirements:
    - Each line: JSON array of messages
    - Each message: {"role": "user"|"assistant", "content": "text"}
    - Roles must alternate starting with "user"
    - Minimum 2 messages per conversation

    Example line:
        [{"role":"user","content":"Hi"},{"role":"assistant","content":"Hello!"}]
    """

    def __init__(self, filepath, **kwargs):
        """
        Initialize the CustomJSON task by loading conversations from a file.

        Args:
            filepath: Path to the JSONL file containing conversations
            **kwargs: Additional arguments passed to parent Task

        The constructor loads all conversations into memory and validates their
        structure to catch formatting errors early.
        """
        super().__init__(**kwargs)
        self.filepath = filepath
        self.conversations = []

        # Check if file exists before trying to load
        if not os.path.exists(filepath):
            # Helpful error message for common case: identity_conversations.jsonl
            # This can be removed in the future once the file is more widely distributed
            print("-" * 80)
            print(f"Warning: File {filepath} does not exist")
            print("HINT (Oct 21 2025)")
            print("If you recently did a git pull and suddely see this, it might be due to the new addition of identity conversations")
            print("See this discussion for more details: https://github.com/karpathy/nanochat/discussions/139")
            print("Quick fix: simply run the following command to download the file and you're done:")
            print(f"curl -L -o {filepath} https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl")
            print("-" * 80)

        else:
            # Load and validate all conversations from the JSONL file
            with open(filepath, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:  # Skip empty lines
                        continue

                    # Parse JSON from this line
                    try:
                        messages = json.loads(line)
                    except json.JSONDecodeError as e:
                        raise ValueError(f"Invalid JSON on line {line_num}: {e}")

                    # Validate conversation structure
                    assert isinstance(messages, list), \
                        f"Line {line_num}: Expected list of messages, got {type(messages)}"
                    assert len(messages) >= 2, \
                        f"Line {line_num}: Conversation must have at least 2 messages, got {len(messages)}"

                    # Validate each message and ensure roles alternate correctly
                    for i, message in enumerate(messages):
                        assert "role" in message, \
                            f"Line {line_num}, message {i}: Missing 'role' field"
                        assert "content" in message, \
                            f"Line {line_num}, message {i}: Missing 'content' field"

                        # Roles must alternate: user, assistant, user, assistant, ...
                        expected_role = "user" if i % 2 == 0 else "assistant"
                        assert message["role"] == expected_role, \
                            f"Line {line_num}, message {i}: Has role '{message['role']}' but should be '{expected_role}'"

                        # Content must be a simple string (not structured parts)
                        assert isinstance(message["content"], str), \
                            f"Line {line_num}, message {i}: Content must be a string, got {type(message['content'])}"

                    # Conversation passed all validation checks
                    self.conversations.append(messages)

        self.length = len(self.conversations)

    def num_examples(self):
        """Returns the total number of conversations loaded from the file."""
        return self.length

    def get_example(self, index):
        """
        Retrieve a single conversation from the loaded data.

        Args:
            index: Index into the conversations list

        Returns:
            dict: Conversation with 'messages' field containing the message list
        """
        messages = self.conversations[index]
        conversation = {
            "messages": messages,
        }
        return conversation

