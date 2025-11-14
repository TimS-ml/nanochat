"""
GSM8K: Grade School Math 8K - A benchmark for math word problem solving.

GSM8K consists of 8,500 linguistically diverse grade school math word problems.
Problems require 2-8 steps to solve and involve basic arithmetic operations.
Solutions are provided with step-by-step reasoning and calculator tool calls.

Dataset: https://huggingface.co/datasets/openai/gsm8k
Paper: "Training Verifiers to Solve Math Word Problems" (Cobbe et al., 2021)

Format:
- Question: Natural language math word problem
- Answer: Step-by-step solution with embedded calculator calls
- Final answer: Numeric value after #### marker

What makes GSM8K challenging:
- Requires multi-step reasoning and planning
- Needs to extract relevant numbers and relationships from natural language
- Must choose appropriate arithmetic operations
- Distractors in the problem text can mislead simple pattern matching
- Requires both mathematical and linguistic understanding

Example problem:
  Question:
  Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of
  babysitting. How much did she earn?

  Answer:
  Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute.
  Working 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10.
  #### 10

Notice the special format:
- Tool calls are wrapped in <<expression=result>> tags (e.g., <<12/60=0.2>>)
- Final answer comes after #### marker
- This format encourages models to show their work and use calculation tools

Evaluation: Generative - extract final numeric answer and compare for exact match.
"""

import re
from datasets import load_dataset
from tasks.common import Task


# Regex pattern to extract the final answer after the #### marker
# Matches: #### followed by optional minus, digits, decimals, and commas
GSM_RE = re.compile(r"#### (\-?[0-9\.\,]+)")

def extract_answer(completion):
    """
    Extract the numerical answer from a GSM8K completion.

    Args:
        completion: Text containing a solution with #### marker

    Returns:
        str: Normalized numeric string, or None if no answer found

    The function looks for the #### marker followed by a number. It normalizes
    the number by removing commas (e.g., "1,234" -> "1234") for consistent
    comparison. This follows the official GSM8K evaluation protocol.

    Examples:
        "... #### 10" -> "10"
        "... #### 1,234.5" -> "1234.5"
        "... #### -42" -> "-42"
        "no marker here" -> None

    Reference:
    https://github.com/openai/grade-school-math/blob/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/dataset.py#L28
    """
    match = GSM_RE.search(completion)
    if match:
        # Extract the matched number string
        match_str = match.group(1).strip()
        # Remove commas for normalization (1,234 -> 1234)
        match_str = match_str.replace(",", "")
        return match_str
    return None


class GSM8K(Task):
    """
    GSM8K (Grade School Math 8K) task for math word problem solving.

    Loads math problems and formats them as conversations with step-by-step
    solutions including calculator tool calls in the <<expr=result>> format.
    """

    def __init__(self, subset, split, **kwargs):
        """
        Initialize the GSM8K dataset.

        Args:
            subset: Either "main" (standard problems) or "socratic" (with more detailed reasoning)
            split: Either "train" (7.5K problems) or "test" (1K problems)
            **kwargs: Additional arguments passed to parent Task
        """
        super().__init__(**kwargs)
        assert subset in ["main", "socratic"], "GSM8K subset must be main|socratic"
        assert split in ["train", "test"], "GSM8K split must be train|test"
        # Load from HuggingFace and shuffle for variety
        self.ds = load_dataset("openai/gsm8k", subset, split=split).shuffle(seed=42)

    @property
    def eval_type(self):
        """GSM8K is a generative task requiring step-by-step math solutions."""
        return 'generative'

    def num_examples(self):
        """Returns the total number of math problems in the dataset."""
        return len(self.ds)

    def get_example(self, index):
        """
        Get a single math problem and format it as a conversation with tool calls.

        Args:
            index: Index into the dataset

        Returns:
            dict: Conversation with:
                - 'messages': User question and assistant solution with tool calls

        The assistant content is structured as a list of parts, alternating between:
        - Text parts: Natural language reasoning steps
        - Python tool calls: Arithmetic expressions (e.g., "12/60")
        - Python outputs: Results of those expressions (e.g., "0.2")

        This multi-part format teaches models to:
        1. Break down problems into steps
        2. Use calculator tools for arithmetic
        3. Show their work clearly
        """
        row = self.ds[index]
        question = row['question']  # The word problem text
        answer = row['answer']  # Solution with <<tool calls>> and #### final answer

        # Parse the answer to extract tool calls and text
        # We need to convert <<expr=result>> into structured tool call parts
        assistant_message_parts = []

        # Split on the <<...>> pattern while preserving the delimiters
        # This gives us alternating text and tool call parts
        parts = re.split(r'(<<[^>]+>>)', answer)

        for part in parts:
            if part.startswith('<<') and part.endswith('>>'):
                # This is a calculator tool call: <<expression=result>>
                inner = part[2:-2]  # Strip the << >> delimiters

                # Split on = to separate expression from result
                if '=' in inner:
                    # Use rsplit to handle cases like "2=2" or "x=y=z" correctly
                    expr, result = inner.rsplit('=', 1)
                else:
                    # Malformed tool call without =, use empty result
                    expr, result = inner, ""

                # Add two parts: the tool call and its output
                assistant_message_parts.append({"type": "python", "text": expr})
                assistant_message_parts.append({"type": "python_output", "text": result})
            else:
                # Regular text between tool calls (reasoning steps)
                assistant_message_parts.append({"type": "text", "text": part})

        # Build the conversation
        messages = [
            {"role": "user", "content": question},  # Simple string
            {"role": "assistant", "content": assistant_message_parts},  # List of parts
        ]

        conversation = {
            "messages": messages,
        }
        return conversation

    def evaluate(self, conversation, assistant_response):
        """
        Evaluate a math solution by comparing the final numeric answer.

        Args:
            conversation: The original problem with ground truth solution
            assistant_response: The model's generated solution (string)

        Returns:
            int: 1 if correct, 0 if incorrect

        Evaluation process:
        1. Extract ground truth answer from conversation (after #### marker)
        2. Extract predicted answer from assistant response (after #### marker)
        3. Normalize both numbers (remove commas)
        4. Compare for exact string match

        Note: We only check the final answer, not the reasoning steps. This means
        the model could use incorrect reasoning but arrive at the right answer by
        luck. Future versions could score intermediate steps for partial credit.

        TODO: Currently assumes assistant_response is a simple string. In the future,
              it could be a list of parts (with tool calls), which would require
              reconstructing the string or extracting from the last text part.
        """
        assert isinstance(assistant_response, str), "Assuming simple string response for now"

        # Extract the ground truth answer from the conversation
        assistant_message = conversation['messages'][-1]
        assert assistant_message['role'] == "assistant", "Last message must be from the Assistant"
        assert isinstance(assistant_message['content'], list), "This is expected to be a list of parts"

        # The final answer is in the last text part (after all tool calls)
        last_text_part = assistant_message['content'][-1]['text']

        # Extract and normalize both answers
        ref_num = extract_answer(last_text_part)  # Ground truth
        pred_num = extract_answer(assistant_response)  # Model's answer

        # Compare the normalized numeric strings
        # Returns 1 for correct, 0 for incorrect
        is_correct = int(pred_num == ref_num)
        return is_correct

    def reward(self, conversation, assistant_response):
        """
        Compute reward for reinforcement learning.

        Args:
            conversation: The original problem
            assistant_response: The model's generated solution

        Returns:
            float: 1.0 if correct, 0.0 if incorrect

        For now, this is just a simple 0/1 reward based on final answer correctness.
        Future versions could provide richer reward signals:
        - Partial credit for correct intermediate steps
        - Penalties for incorrect tool usage
        - Bonuses for efficient solutions
        - Format matching rewards (using #### marker correctly, etc.)
        """
        is_correct = self.evaluate(conversation, assistant_response)
        is_correct_float = float(is_correct)
        return is_correct_float
