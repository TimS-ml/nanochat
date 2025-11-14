"""
The MMLU (Massive Multitask Language Understanding) benchmark.

MMLU is a comprehensive benchmark for measuring knowledge and reasoning across 57 subjects
spanning STEM, humanities, social sciences, and more. It's designed to test both breadth
and depth of a model's understanding across diverse academic domains.

Dataset: https://huggingface.co/datasets/cais/mmlu
Paper: "Measuring Massive Multitask Language Understanding" (Hendrycks et al., 2021)

Coverage includes:
- STEM: mathematics, physics, chemistry, biology, computer science
- Humanities: history, philosophy, law
- Social Sciences: economics, psychology, politics
- Other: professional medicine, business ethics, etc.

What makes MMLU challenging:
- Requires factual knowledge across 57 diverse subjects
- Questions range from high school to professional level difficulty
- Tests both memorization and reasoning capabilities
- Some questions require multi-step reasoning or domain-specific expertise
- Performance on MMLU is often used as a proxy for general knowledge/intelligence

Format: 4-way multiple choice (A, B, C, D)
Evaluation: Categorical - model must output the correct letter

The dataset has two main subsets:
- "all": Contains validation, dev, and test splits across all 57 subjects
- "auxiliary_train": Additional training data (note: has quirky nested structure)
"""

from datasets import load_dataset
from tasks.common import Task, render_mc

class MMLU(Task):
    """
    MMLU (Massive Multitask Language Understanding) benchmark task.

    Loads and formats MMLU questions as multiple choice conversations.
    MMLU always uses 4-way multiple choice with letters A, B, C, D.
    """

    # MMLU always uses 4-way multiple choice
    letters = ('A', 'B', 'C', 'D')

    # All 57 subject groups in MMLU, spanning diverse academic domains
    groups = ('abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge',
              'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics',
              'college_medicine', 'college_physics', 'computer_security', 'conceptual_physics',
              'econometrics', 'electrical_engineering', 'elementary_mathematics', 'formal_logic',
              'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science',
              'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics',
              'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics',
              'high_school_physics', 'high_school_psychology', 'high_school_statistics',
              'high_school_us_history', 'high_school_world_history', 'human_aging', 'human_sexuality',
              'international_law', 'jurisprudence', 'logical_fallacies', 'machine_learning',
              'management', 'marketing', 'medical_genetics', 'miscellaneous', 'moral_disputes',
              'moral_scenarios', 'nutrition', 'philosophy', 'prehistory', 'professional_accounting',
              'professional_law', 'professional_medicine', 'professional_psychology', 'public_relations',
              'security_studies', 'sociology', 'us_foreign_policy', 'virology', 'world_religions')

    def __init__(self, subset, split, **kwargs):
        """
        Initialize the MMLU dataset.

        Args:
            subset: Either "all" (evaluation) or "auxiliary_train" (training)
            split: One of "train", "validation", "dev", or "test"
            **kwargs: Additional arguments passed to parent Task
        """
        super().__init__(**kwargs)
        assert subset in ["all", "auxiliary_train"], f"subset {subset} must be all|auxiliary_train"
        assert split in ["train", "validation", "dev", "test"], f"split {split} must be train|validation|dev|test"
        # auxiliary_train only has training data
        if subset == "auxiliary_train":
            assert split == "train", "auxiliary_train must be split into train"

        self.subset = subset
        self.split = split
        # Load from HuggingFace and shuffle with fixed seed for reproducibility
        self.ds = load_dataset("cais/mmlu", subset, split=split).shuffle(seed=42)

        # Handle quirky data structure in auxiliary_train subset
        # For some reason, auxiliary_train wraps each row in an extra 'train' field
        # We unwrap it here to get the actual question data
        if subset == "auxiliary_train":
            self.ds = self.ds.map(lambda row: row['train'], remove_columns=['train'])

    @property
    def eval_type(self):
        """MMLU is a categorical (multiple choice) task."""
        return 'categorical'

    def num_examples(self):
        """Returns the total number of questions in the dataset."""
        return len(self.ds)

    def get_example(self, index):
        """
        Retrieve a single MMLU question and format it as a conversation.

        Args:
            index: Index into the dataset

        Returns:
            dict: Conversation with:
                - 'messages': List with user question and assistant answer
                - 'subject': Subject area (e.g., "college_biology")
                - 'letters': Valid answer letters for evaluation clamping
        """
        row = self.ds[index]
        # Extract question components from the dataset row
        question = row["question"]  # The question text
        choices = row["choices"]  # List of 4 answer choice texts
        answer = row["answer"]  # Integer index of correct answer (0, 1, 2, or 3)
        subject = row["subject"]  # Subject category (e.g., "college_biology")

        # Sanity check: MMLU always has exactly 4 choices
        assert len(choices) == 4, "MMLU should have 4 choices"

        # Format the question using the standard multiple choice renderer
        user_message = render_mc(question, self.letters, choices)

        # Convert answer index (0-3) to letter (A-D)
        assistant_message = self.letters[answer]

        # Build the conversation
        messages = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message}
        ]

        # Return conversation with metadata
        conversation = {
            "messages": messages,
            # Subject is useful for computing per-category metrics
            # (e.g., "How well does the model do on STEM vs humanities?")
            "subject": subject,
            # Store valid letters for evaluation clamping
            "letters": self.letters,
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
        of the valid letters A-D by the evaluation code.
        """
        # Sanity check: response should be one of A, B, C, D
        assert assistant_response in self.letters, \
            f"MMLU answer {assistant_response} is expected to be one of {self.letters}"

        # Extract the ground truth answer from the conversation
        assistant_message = conversation['messages'][-1]['content']  # e.g., "A"

        # Simple exact match comparison
        return assistant_response == assistant_message
