"""
HumanEval: Evaluating Large Language Models on Code Generation.

Despite the name, HumanEval has nothing to do with human evaluation - it's an automated
coding benchmark consisting of 164 Python programming problems. Each problem provides a
function signature and docstring, and the model must generate the function body.

Dataset: https://huggingface.co/datasets/openai/openai_humaneval
Paper: "Evaluating Large Language Models Trained on Code" (Chen et al., 2021)

Format:
- Input: Function signature + docstring describing the task
- Output: Complete function implementation
- Evaluation: Functional correctness via unit tests (pass@k metric)

What makes HumanEval challenging:
- Requires understanding natural language specifications
- Must translate requirements into working code
- Needs to handle edge cases and corner cases
- Tests are hidden from the model, so it must infer test cases from the description
- Problems range from simple string manipulation to complex algorithms

Example problem:
    def has_close_elements(numbers: List[float], threshold: float) -> bool:
        \"\"\" Check if in given list of numbers, are any two numbers closer to each other than
        given threshold.
        >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
        False
        >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
        True
        \"\"\"
        # Model must generate implementation here

Evaluation: Generative task - code is executed and tested for correctness.
The standard metric is pass@k (percentage of problems solved with k samples).
"""

import re
from datasets import load_dataset
from nanochat.execution import execute_code
from tasks.common import Task

def extract_imports(prompt):
    """
    Extract import statements from the beginning of a code block.

    Args:
        prompt: Python code string

    Returns:
        str: Newline-joined import statements

    HumanEval prompts often include necessary imports at the top (e.g., typing.List).
    We extract these to prepend them to the model's generated code, since the model
    might generate just the function body without repeating the imports.

    The extraction stops at the first non-import, non-comment line to avoid
    capturing function definitions or other code.
    """
    imports = []
    for line in prompt.split('\n'):
        stripped = line.strip()
        if stripped.startswith('import ') or stripped.startswith('from '):
            imports.append(stripped)
        elif stripped and not stripped.startswith('#'):
            # Stop at first non-import, non-comment line
            # (likely the function definition)
            break
    return '\n'.join(imports)

def extract_program(completion):
    """
    Extract Python code from LLM completion, handling various markdown formats.

    Args:
        completion: Raw text output from the language model

    Returns:
        str: Extracted Python code

    Models often wrap code in markdown fences, especially chat-tuned models.
    This function handles common formats:
    - ```python\\n...\\n``` - Standard Python code block
    - ```\\n...\\n``` - Generic code block
    - Plain code without any markdown

    Strategy:
    1. Look for markdown code blocks first (most common with chat models)
    2. If found, extract the first code block (ignore any explanatory text)
    3. If no code blocks, assume the entire completion is code

    This robustness is necessary because different models have different formatting
    conventions, and we want evaluation to work across all of them.
    """
    # Try to find markdown code blocks
    # Pattern matches ```python or ``` followed by code followed by ```
    # The (?:python)? makes "python" optional and non-capturing
    # re.DOTALL makes . match newlines too
    pattern = r'```(?:python)?\s*\n(.*?)\n```'
    matches = re.findall(pattern, completion, re.DOTALL)

    if matches:
        # Found markdown-wrapped code, return the first block
        # (ignore any explanation text that might come after)
        return matches[0].strip()

    # No markdown blocks found, assume the entire completion is code
    # This handles models that output plain code without formatting
    return completion.strip()

class HumanEval(Task):
    """
    HumanEval coding benchmark task.

    Loads Python programming problems and evaluates solutions via unit test execution.
    Each problem consists of a function signature with docstring, and the model must
    generate the implementation.
    """

    def __init__(self, **kwargs):
        """
        Initialize the HumanEval dataset.

        Args:
            **kwargs: Additional arguments passed to parent Task

        Note: HumanEval only has a test split (164 problems), no train/validation.
        """
        super().__init__(**kwargs)
        # Load the dataset and shuffle for variety
        self.ds = load_dataset("openai/openai_humaneval", split="test").shuffle(seed=42)

    @property
    def eval_type(self):
        """HumanEval is a generative task requiring code generation."""
        return 'generative'

    def num_examples(self):
        """Returns the total number of coding problems (164)."""
        return len(self.ds)

    def get_example(self, index):
        """
        Get a single coding problem from the dataset.

        Args:
            index: Index into the dataset

        Returns:
            dict: Conversation with:
                - 'messages': User prompt and assistant solution
                - 'entry_point': Function name to test
                - 'test': Unit test code for verification

        The prompt contains the function signature and docstring. The solution is the
        canonical (correct) implementation. During evaluation, we'll test the model's
        generated code against the provided unit tests.
        """
        row = self.ds[index]
        # Extract problem components
        prompt = row['prompt']  # Function signature + docstring (what the model sees)
        solution = row['canonical_solution']  # Correct implementation (reference answer)
        entry_point = row['entry_point']  # Function name (e.g., "has_close_elements")
        test = row['test']  # Unit test code with assertions

        # The complete solution is prompt + implementation
        # This is what we'd want the model to generate
        complete_solution = f"{prompt}\n{solution}"

        # Build conversation
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": complete_solution},
        ]

        # Return with metadata needed for evaluation
        conversation = {
            "messages": messages,
            "entry_point": entry_point,  # Function name to call in tests
            "test": test,  # Test cases to verify correctness
        }
        return conversation

    def evaluate(self, conversation, completion):
        """
        Evaluate a code completion by executing it against unit tests.

        Args:
            conversation: The original problem from the dataset
            completion: The model's generated code (may be raw or markdown-wrapped)

        Returns:
            bool: True if all tests pass, False if any test fails or execution errors occur

        Evaluation process:
        1. Extract imports from the original prompt (typing hints, etc.)
        2. Extract code from the completion (handle markdown formatting)
        3. Combine into executable program: imports + code + tests + check call
        4. Execute in sandboxed environment
        5. Return success based on whether all assertions pass

        The test execution uses the `check()` function pattern from HumanEval, which
        runs all the test cases and raises an assertion error if any fail.
        """
        # Extract imports from the prompt (e.g., "from typing import List")
        # These are needed for type hints but might not be in the model's output
        imports = extract_imports(conversation['messages'][0]['content'])

        # Extract the actual code from the completion
        # This handles markdown formatting that chat models often use
        completion_code = extract_program(completion)

        # Assemble the complete program to execute
        # Structure: imports, model's code, test definitions, check call
        program = (
            imports  # Type hints and other necessary imports
            + "\n\n"
            + completion_code  # The model's generated function
            + "\n\n"
            + conversation['test']  # Test case definitions (def check(candidate):...)
            + "\n"
            + f"check({conversation['entry_point']})"  # Call check with the function
        )

        # Execute the program in a sandboxed environment
        # This runs the tests and captures success/failure
        result = execute_code(program)
        success = result.success

        return success
