"""
Base class for all Tasks.

A Task represents a dataset of conversations, together with metadata and evaluation criteria.
This module provides the foundation for all benchmark and training tasks in nanochat.

Tasks serve two main purposes:
1. Training: Provide conversational data for supervised fine-tuning (SFT) or reinforcement learning (RL)
2. Evaluation: Assess model performance on specific benchmarks with well-defined metrics

The Task class supports lightweight slicing using start/stop/step parameters, allowing you to
work with subsets of data without copying the underlying dataset.

Example tasks include:
- MMLU: Multiple-choice questions across 57 academic subjects
- ARC: Science questions at elementary and middle school level
- GSM8K: Grade school math word problems
- HumanEval: Python programming challenges
- SmolTalk: General conversational data for chat capabilities
"""

import random

class Task:
    """
    Base class of a Task. Allows for lightweight slicing of the underlying dataset.

    This class provides a common interface for all tasks and supports Python-style
    slicing without copying data. Subclasses must implement:
    - eval_type: Returns 'generative' or 'categorical' to indicate evaluation mode
    - num_examples(): Returns total number of examples in the dataset
    - get_example(index): Returns a conversation dict for the given index
    - evaluate(conversation, completion): Evaluates model output and returns success metric
    """

    def __init__(self, start=0, stop=None, step=1):
        """
        Initialize a Task with optional slicing parameters.

        Args:
            start: Index to start from (inclusive), defaults to 0
            stop: Index to stop at (exclusive), defaults to None (end of dataset)
            step: Step size for striding through data, defaults to 1

        The slicing allows creating lightweight views over datasets, similar to
        Python's range(start, stop, step). For example, Task(start=10, stop=20, step=2)
        would give you indices [10, 12, 14, 16, 18] from the underlying dataset.
        """
        # Validate slicing parameters to prevent common errors
        assert start >= 0, f"Start must be non-negative, got {start}"
        assert stop is None or stop >= start, f"Stop should be greater than or equal to start, got {stop} and {start}"
        assert step >= 1, f"Step must be strictly positive, got {step}"
        self.start = start
        self.stop = stop  # None means "until end of dataset"
        self.step = step

    @property
    def eval_type(self):
        """
        Returns the evaluation type for this task.

        Returns:
            str: Either 'generative' or 'categorical'
            - 'categorical': Multiple choice tasks where answer is selected from fixed options (e.g., MMLU, ARC)
            - 'generative': Open-ended tasks requiring generated text (e.g., GSM8K, HumanEval)
        """
        raise NotImplementedError

    def num_examples(self):
        """
        Returns the total number of examples in the underlying dataset.

        Returns:
            int: Total count of examples before any slicing is applied
        """
        raise NotImplementedError

    def get_example(self, index):
        """
        Retrieve a single example from the dataset at the given physical index.

        Args:
            index: Physical index into the underlying dataset (not the logical sliced index)

        Returns:
            dict: A conversation dictionary containing:
                - 'messages': List of message dicts with 'role' and 'content' fields
                - Additional task-specific metadata (e.g., 'letters' for multiple choice)
        """
        raise NotImplementedError

    def __len__(self):
        """
        Returns the number of examples in the sliced view of the dataset.

        Calculates the logical length based on start/stop/step parameters.
        For example, if the dataset has 100 items and we have start=10, stop=30, step=2,
        then len() returns 10 (items at indices 10, 12, 14, ..., 28).

        Returns:
            int: Number of examples accessible through this Task view
        """
        start = self.start
        stop = self.num_examples() if self.stop is None else self.stop
        step = self.step
        span = stop - start
        # Calculate ceiling division to get the number of steps
        # Example: span=19, step=2 gives (19+2-1)//2 = 10
        num = (span + step - 1) // step  # ceil_div(span, step)
        assert num >= 0, f"Negative number of examples???: {num}"  # Sanity check
        return num

    def __getitem__(self, index: int):
        """
        Access a conversation at the given logical index in the sliced view.

        Args:
            index: Logical index within the sliced view (0-based)

        Returns:
            dict: Conversation object from get_example()

        This method translates the logical index to a physical index in the underlying
        dataset. For example, if start=10 and step=2, then logical index 0 maps to
        physical index 10, logical index 1 maps to physical index 12, etc.
        """
        assert isinstance(index, int), f"Index must be an integer, got {type(index)}"
        # Map logical sliced index to physical dataset index
        physical_index = self.start + index * self.step
        conversation = self.get_example(physical_index)
        return conversation

    def evaluate(self, problem, completion):
        """
        Evaluate model completion for a given problem.

        Args:
            problem: The conversation/problem from the dataset
            completion: The model's generated response

        Returns:
            Metric value indicating correctness (typically bool or float)
            - For categorical tasks: True/False for correct/incorrect
            - For generative tasks: Could be bool, int, or float depending on task
        """
        raise NotImplementedError


class TaskMixture(Task):
    """
    Combines multiple tasks into a single shuffled dataset for training.

    For supervised fine-tuning (SFT), it's beneficial to train on a mixture of diverse datasets
    to develop broad capabilities. This class takes multiple Task objects and shuffles their
    examples together deterministically.

    Key features:
    - Deterministic shuffling (seed=42) ensures reproducibility across runs
    - Uniform mixing prevents task clustering that could hurt optimization
    - Simple oversampling: just pass the same task multiple times to increase its proportion

    Example:
        # Mix MMLU, ARC, and double-weight GSM8K
        mixture = TaskMixture([mmlu_task, arc_task, gsm8k_task, gsm8k_task])
    """

    def __init__(self, tasks, **kwargs):
        """
        Initialize a mixture of tasks with deterministic shuffling.

        Args:
            tasks: List of Task objects to combine
            **kwargs: Passed to parent Task class for slicing support
        """
        super().__init__(**kwargs)
        self.tasks = tasks
        self.lengths = [len(task) for task in self.tasks]
        self.num_conversations = sum(self.lengths)

        # Build a mapping from global index to (task_idx, local_idx) pairs
        # This allows us to efficiently look up which task and which example
        # within that task corresponds to any global index
        self.index_map = []
        for task_idx, task_length in enumerate(self.lengths):
            for local_idx in range(task_length):
                self.index_map.append((task_idx, local_idx))

        # Shuffle the index map deterministically to mix tasks throughout training
        # This prevents having all examples from one task grouped together, which
        # would create uneven gradients and hurt optimization
        rng = random.Random(42)
        rng.shuffle(self.index_map)
        # Note: This creates an in-memory list of all indices, which is fine for
        # most use cases but could be optimized for very large datasets

    def num_examples(self):
        """Returns total number of conversations across all tasks in the mixture."""
        return self.num_conversations

    def get_example(self, index):
        """
        Access conversations according to a deterministic shuffle of all examples.

        Args:
            index: Global index into the shuffled mixture

        Returns:
            dict: Conversation from the appropriate task

        The shuffling ensures tasks are mixed throughout training, preventing
        catastrophic forgetting and enabling better multi-task learning.
        """
        assert 0 <= index < self.num_conversations, f"Index {index} out of range for mixture with {self.num_conversations} conversations"
        # Look up which task and local index this global index maps to
        task_idx, local_idx = self.index_map[index]
        return self.tasks[task_idx][local_idx]


class TaskSequence(Task):
    """
    Concatenates multiple tasks in sequential order without shuffling.

    Unlike TaskMixture which shuffles tasks together, TaskSequence presents tasks
    one after another in the order provided. This is useful for curriculum learning
    where you want to train on tasks in a specific order (e.g., easier tasks first).

    The sequence simply concatenates tasks: if task A has 100 examples and task B
    has 50 examples, indices 0-99 come from A and indices 100-149 come from B.

    Example:
        # Train on simple tasks before complex ones
        sequence = TaskSequence([simple_task, medium_task, hard_task])
    """

    def __init__(self, tasks, **kwargs):
        """
        Initialize a sequential concatenation of tasks.

        Args:
            tasks: List of Task objects to concatenate in order
            **kwargs: Passed to parent Task class for slicing support
        """
        super().__init__(**kwargs)
        self.tasks = tasks
        self.lengths = [len(task) for task in self.tasks]
        self.num_conversations = sum(self.lengths)

    def num_examples(self):
        """Returns total number of conversations across all tasks in sequence."""
        return self.num_conversations

    def get_example(self, index):
        """
        Access conversations sequentially from concatenated tasks.

        Args:
            index: Global index into the sequence

        Returns:
            dict: Conversation from the appropriate task

        This method finds which task contains the requested index by subtracting
        task lengths until we find the right task, then returns that example.
        """
        assert 0 <= index < self.num_conversations, f"Index {index} out of range for sequence with {self.num_conversations} conversations"
        # Iterate through tasks to find which one contains this index
        for task_idx, task_length in enumerate(self.lengths):
            if index < task_length:
                # Found the right task, return the example at the local index
                return self.tasks[task_idx][index]
            # Not in this task, subtract its length and continue
            index -= task_length


def render_mc(question, letters, choices):
    """
    Render a multiple choice question in a standardized format optimized for small models.

    Args:
        question: The question text
        letters: List of letter labels (e.g., ['A', 'B', 'C', 'D'])
        choices: List of answer choice texts corresponding to each letter

    Returns:
        str: Formatted prompt string for the model

    Format example:
        Multiple Choice question: What is the capital of France?
        - Paris=A
        - London=B
        - Berlin=C
        - Madrid=D

        Respond only with the letter of the correct answer.

    Two critical design decisions for small model performance:

    1) Letter placement: We put the letter AFTER the choice text (choice=letter)
       rather than before (letter. choice). This improves binding for smaller models
       because the model sees the content first, then associates it with the letter.
       Larger models are less sensitive to this ordering.

    2) No whitespace before letters: We use "=A" not "= A" or " A".
       This is crucial because tokenizers treat " A" and "A" as different tokens.
       Since the assistant response will be just "A" (no leading space), we need
       the prompt to use the same token "A" for consistent next-token prediction.
       Small models are very sensitive to these tokenization details.
    """
    query = f"Multiple Choice question: {question}\n"
    # Format each choice as "- choice_text=letter\n"
    query += "".join([f"- {choice}={letter}\n" for letter, choice in zip(letters, choices)])
    query += "\nRespond only with the letter of the correct answer."
    return query


if __name__ == "__main__":
    # very lightweight test of slicing
    from tasks.mmlu import MMLU

    ds = MMLU(subset="auxiliary_train", split="train")
    print("Length of MMLU: ", len(ds))
    ex = ds[5]
    print("5th example: ", ex)

    ds = MMLU(subset="auxiliary_train", split="train", start=5, stop=10)
    print("Length of sliced MMLU[5:10]: ", len(ds))
    print("0th example of sliced MMLU: ", ds[0])

    print("They match: ", ex == ds[0])
