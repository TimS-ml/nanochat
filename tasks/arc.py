"""
The ARC (AI2 Reasoning Challenge) dataset from Allen AI.

ARC is a multiple-choice question answering benchmark consisting of science questions
at the elementary and middle school level. The dataset is designed to test reasoning
and scientific knowledge, with questions derived from actual standardized tests.

Dataset: https://huggingface.co/datasets/allenai/ai2_arc
Paper: "Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge" (2018)

Two difficulty subsets:
- ARC-Easy: Questions that can often be answered with simple fact retrieval or word overlap
- ARC-Challenge: Questions requiring deeper reasoning, multi-hop inference, or domain knowledge

What makes ARC challenging:
- Requires real scientific knowledge, not just pattern matching
- Questions designed to be difficult for retrieval-based methods
- Many questions require combining multiple facts or concepts
- Distractors (wrong answers) are carefully crafted to be plausible

Example question:
  Q: Which property of a mineral can be determined just by looking at it?
  A: color     B: hardness     C: luster     D: mass
  Correct: A (though luster could also be argued - these edge cases add difficulty)

Evaluation: Categorical (multiple choice) - model must output the correct letter.
"""

from datasets import load_dataset
from tasks.common import Task, render_mc

class ARC(Task):
    """
    ARC (AI2 Reasoning Challenge) task for science question answering.

    Loads and formats ARC questions as multiple choice conversations,
    using the standard render_mc format for consistency across tasks.
    """

    def __init__(self, subset, split, **kwargs):
        """
        Initialize the ARC dataset.

        Args:
            subset: Either "ARC-Easy" or "ARC-Challenge"
            split: One of "train", "validation", or "test"
            **kwargs: Additional arguments passed to parent Task (e.g., slicing parameters)
        """
        super().__init__(**kwargs)
        assert subset in ["ARC-Easy", "ARC-Challenge"], "ARC subset must be ARC-Easy or ARC-Challenge"
        assert split in ["train", "validation", "test"], "ARC split must be train|validation|test"
        # Load from HuggingFace and shuffle with fixed seed for reproducibility
        self.ds = load_dataset("allenai/ai2_arc", subset, split=split).shuffle(seed=42)

    @property
    def eval_type(self):
        """ARC is a categorical (multiple choice) task."""
        return 'categorical'

    def num_examples(self):
        """Returns the total number of questions in the dataset."""
        return len(self.ds)

    def get_example(self, index):
        """
        Retrieve a single ARC question and format it as a conversation.

        Args:
            index: Index into the dataset

        Returns:
            dict: Conversation with:
                - 'messages': List with user question and assistant answer
                - 'letters': Valid answer letters for evaluation clamping
        """
        row = self.ds[index]
        # Extract question components from the dataset row
        question = row["question"]  # The question text
        choices = row["choices"]["text"]  # List of answer choice texts
        answer_string = row["answerKey"]  # Correct answer letter (e.g., "A", "B", "C", "D")
        letters = row["choices"]["label"]  # List of all choice letters

        # Validate that the answer key is actually one of the provided letters
        assert answer_string in letters, f"ARC answer {answer_string} must be one of {letters}"

        # Format the question using the standard multiple choice renderer
        # This ensures consistent formatting across all multiple choice tasks
        user_message = render_mc(question, letters, choices)

        # Build the conversation with user question and correct answer
        messages = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": answer_string}  # Just the letter, e.g., "A"
        ]

        # Return conversation with metadata needed for evaluation
        conversation = {
            "messages": messages,
            # Store valid letters for evaluation - allows us to clamp model predictions
            # to valid choices even if it generates extra text
            "letters": letters,
        }
        return conversation

    def evaluate(self, conversation, assistant_response):
        """
        Evaluate whether the model's response matches the correct answer.

        Args:
            conversation: The original conversation dict with ground truth
            assistant_response: The model's generated response (should be a single letter)

        Returns:
            bool: True if the response matches the correct answer, False otherwise

        The evaluation expects that assistant_response has already been clamped to one
        of the valid letters (this happens in the evaluation loop). We just compare it
        to the ground truth answer from the conversation.
        """
        # Sanity check: response should be clamped to valid letters by evaluation code
        # This assertion helps catch bugs in the evaluation pipeline
        assert assistant_response in conversation['letters'], \
            f"ARC answer {assistant_response} is expected to be one of {conversation['letters']}"

        # Extract the ground truth answer from the conversation
        assistant_message = conversation['messages'][-1]['content']  # e.g., "A"

        # Simple exact match comparison
        return assistant_response == assistant_message
