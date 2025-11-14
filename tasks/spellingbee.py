"""
Task intended to make nanochat better in spelling and counting, for example:

"How many r are in strawberry?" -> 3

An interesting part of this task is that we will get the assistant to
solve the problem using a combination of manual counting and Python.
This is a good problem solving "instinct" to mix into the model and RL
may further refine it to trust one over the other. If we were extra fancy
(which we could/should be) we'd add small errors here and there to allow
the model also learn recoveries. We can do this in future versions.

There are two tasks in this file:
1. SpellingBee: Counting the number of occurrences of a letter in a word
2. SimpleSpelling: Simply spelling words

(1) is the goal, but (2) exists as a highly condensed version of the part
that makes (1) difficult, which is word spelling. This is non-trivial for an
LLM because it has to learn how every token (a little semantic chunk/atom)
maps to the sequence of individual characters that make it up. Larger models
learn this eventually on their own, but if we want this capability to exist
in smaller models, we have to actively encourage it by over-representing it
in the training data. Midtraining is a good place to do this.

To preview a few example conversations, run:
python -m tasks.spellingbee
"""

import re
import random
from tasks.common import Task
from nanochat.common import download_file_with_lock

# Letters of the alphabet
LETTERS = "abcdefghijklmnopqrstuvwxyz"
# A list of 370K English words of large variety
WORD_LIST_URL = "https://raw.githubusercontent.com/dwyl/english-words/refs/heads/master/words_alpha.txt"

# Regex pattern to extract final answer after #### marker (identical to GSM8K)
ANSWER_RE = re.compile(r"#### (\-?[0-9\.\,]+)")

def extract_answer(completion):
    """
    Extract the numerical answer after the #### marker.

    Args:
        completion: Text containing a solution with #### marker

    Returns:
        str: Normalized numeric string, or None if no answer found

    This is identical to GSM8K's answer extraction - we use the same format
    for consistency across math-style tasks.
    """
    match = ANSWER_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")  # Normalize: remove commas
        return match_str
    return None

# User message templates for data augmentation
USER_MSG_TEMPLATES = [
    "How many {letter} are in the word {word}",
    "How many {letter} are in {word}",
    "Count the number of {letter} in {word}",
    "How many times does {letter} appear in {word}",
    "What's the count of {letter} in {word}",
    "In the word {word}, how many {letter} are there",
    "How many letter {letter} are in the word {word}",
    "Count how many {letter} appear in {word}",
    "Tell me the number of {letter} in {word}",
    "How many occurrences of {letter} are in {word}",
    "Find the count of {letter} in {word}",
    "Can you count the {letter} letters in {word}",
    "What is the frequency of {letter} in {word}",
    "How many {letter}s are in {word}",
    "How many {letter}'s are in {word}",
    "Count all the {letter} in {word}",
    "How many times is {letter} in {word}",
    "Number of {letter} in {word}",
    "Total count of {letter} in {word}",
    "How many {letter} does {word} have",
    "How many {letter} does {word} contain",
    "What's the number of {letter} in {word}",
    "{word} has how many {letter}",
    "In {word}, count the {letter}",
    "How many {letter} appear in {word}",
    "Count the {letter} in {word}",
    "Give me the count of {letter} in {word}",
    "How many instances of {letter} in {word}",
    "Show me how many {letter} are in {word}",
    "Calculate the number of {letter} in {word}",
    # Spanish
    "¿Cuántas {letter} hay en {word}?",
    "¿Cuántas veces aparece {letter} en {word}?",
    "Cuenta las {letter} en {word}",
    "¿Cuántas letras {letter} tiene {word}?",
    # Chinese (Simplified)
    "{word}中有多少个{letter}",
    "{word}里有几个{letter}",
    "数一下{word}中的{letter}",
    "{word}这个词里有多少{letter}",
    # Korean
    "{word}에 {letter}가 몇 개 있나요",
    "{word}에서 {letter}의 개수는",
    "{word}에 {letter}가 몇 번 나오나요",
    "{word}라는 단어에 {letter}가 몇 개",
    # French
    "Combien de {letter} dans {word}",
    "Combien de fois {letter} apparaît dans {word}",
    "Compte les {letter} dans {word}",
    # German
    "Wie viele {letter} sind in {word}",
    "Wie oft kommt {letter} in {word} vor",
    "Zähle die {letter} in {word}",
    # Japanese
    "{word}に{letter}は何個ありますか",
    "{word}の中に{letter}がいくつ",
    "{word}に{letter}が何回出てくる",
]

class SpellingBee(Task):
    """
    SpellingBee: Counting letter occurrences in words.

    This task teaches models to count how many times a specific letter appears
    in a word. It's designed to improve small models' spelling and character-level
    reasoning by combining manual counting with Python verification.

    The task is challenging for LLMs because:
    - Models see words as tokens (semantic chunks), not character sequences
    - Requires understanding how tokens decompose into individual characters
    - Small models often struggle with this without explicit training

    Example problem:
        Q: How many 'r' are in the word 'strawberry'?
        A: Manual spelling + counting + Python verification -> #### 3
    """

    def __init__(self, size=1000, split="train", **kwargs):
        """
        Initialize the SpellingBee task with synthetic problems.

        Args:
            size: Number of synthetic problems to generate
            split: Either "train" or "test" (uses different random seeds)
            **kwargs: Additional arguments passed to parent Task

        The task generates problems on-the-fly from a large word list (~370K words).
        Different splits use different random seeds to ensure no overlap.
        """
        super().__init__(**kwargs)
        assert split in ["train", "test"], "SpellingBee split must be train|test"
        self.size = size
        self.split = split

        # Download the English word list if not already cached
        filename = WORD_LIST_URL.split("/")[-1]
        word_list_path = download_file_with_lock(WORD_LIST_URL, filename)

        # Load all words from the file
        with open(word_list_path, 'r', encoding='utf-8') as f:
            words = [line.strip() for line in f]
        self.words = words  # ~370K English words

    @property
    def eval_type(self):
        """SpellingBee is a generative task with numeric answers."""
        return 'generative'

    def num_examples(self):
        """Returns the configured size (number of synthetic problems)."""
        return self.size

    def get_example(self, index):
        """
        Generate a synthetic letter-counting problem.

        Args:
            index: Index determining which problem to generate

        Returns:
            dict: Conversation with step-by-step solution using tool calls

        The problem is generated deterministically from the index using a random seed.
        This ensures reproducibility while creating diverse problems across the dataset.
        Train and test splits use different seeds to avoid overlap.
        """
        # Use index as random seed, but negate for test split to avoid overlap
        # This ensures train[0] != test[0] even though both use index 0
        seed = index if self.split == "train" else -(index + 1)
        rng = random.Random(seed)

        # Pick a random word from the word list
        word = rng.choice(self.words)

        # Pick a letter to count:
        # - 90% of the time: pick a letter from the word (ensures non-zero counts are common)
        # - 10% of the time: pick a random letter (allows for zero counts)
        # This distribution makes the task more interesting than always picking present letters
        letter = rng.choice(word) if rng.random() < 0.9 else rng.choice(LETTERS)

        # Compute the correct answer by counting occurrences
        count = word.count(letter)

        # Create user message with heavy data augmentation for robustness
        # We want the model to work with diverse phrasings and formatting

        # Choose a random template from 30+ variations across multiple languages
        template = rng.choice(USER_MSG_TEMPLATES)

        # 30% chance to lowercase the entire template
        # Simulates casual users who don't capitalize
        if rng.random() < 0.3:
            template = template.lower()

        # Randomly decide whether to quote the letter and word
        # This creates variety: 'r' in strawberry, r in "strawberry", etc.
        quote_options = ['', "'", '"']
        letter_quote = rng.choice(quote_options)
        word_quote = rng.choice(quote_options)
        letter_wrapped = f"{letter_quote}{letter}{letter_quote}"
        word_wrapped = f"{word_quote}{word}{word_quote}"

        # Fill in the template with the wrapped letter and word
        user_msg = template.format(letter=letter_wrapped, word=word_wrapped)

        # 50% chance to add a question mark at the end
        # Many users forget punctuation, so we want robustness to both
        if rng.random() < 0.5:
            user_msg += "?"

        # Build the ideal assistant response as a list of parts
        # The solution demonstrates a two-phase approach:
        # 1. Manual character-by-character counting (shows reasoning)
        # 2. Python verification (teaches tool usage)
        assistant_parts = []

        # Phase 1: Manual counting with detailed work shown
        word_letters = ",".join(list(word))  # e.g., "s,t,r,a,w,b,e,r,r,y"

        manual_text = f"""We are asked to find the number '{letter}' in the word '{word}'. Let me try a manual approach first.

First spell the word out:
{word}:{word_letters}

Then count the occurrences of '{letter}':
"""

        # Simulate the counting process step by step
        # This teaches the model to:
        # - Spell words character by character
        # - Track a running count
        # - Show their work
        #
        # TODO: Future enhancement - occasionally introduce small mistakes that the
        # model catches and corrects. This would teach error recovery, which could
        # also emerge from RL but would benefit from explicit supervision.
        running_count = 0
        for i, char in enumerate(word, 1):
            if char == letter:
                running_count += 1
                # CRITICAL: No space between i and char (f"{i}:{char}")
                # This is because " a" and "a" are different tokens!
                # We want "a" to appear as a single token both here and in the word
                manual_text += f"{i}:{char} hit! count={running_count}\n"
            else:
                manual_text += f"{i}:{char}\n"

        manual_text += f"\nThis gives us {running_count}."
        assistant_parts.append({"type": "text", "text": manual_text})

        # Phase 2: Python verification
        # This teaches the model to verify manual work with code
        assistant_parts.append({"type": "text", "text": "\n\nLet me double check this using Python:\n\n"})

        # Python tool call: word.count(letter)
        python_expr = f"'{word}'.count('{letter}')"
        assistant_parts.append({"type": "python", "text": python_expr})

        # Python output: the actual count
        assistant_parts.append({"type": "python_output", "text": str(count)})

        # Final answer in standard #### format (like GSM8K)
        assistant_parts.append({"type": "text", "text": f"\n\nPython gives us {count}.\n\nMy final answer is:\n\n#### {count}"})

        # return the full conversation
        messages = [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_parts}
        ]
        conversation = {
            "messages": messages,
        }
        return conversation

    def evaluate(self, conversation, assistant_response):
        """
        Given (conversation, completion), return evaluation outcome (0 = wrong, 1 = correct)
        Identical to gsm8k's evaluation.
        """
        assert isinstance(assistant_response, str), "Assuming simple string response for now"
        # First extract the ground truth answer from the conversation
        assistant_message = conversation['messages'][-1]
        assert assistant_message['role'] == "assistant", "Last message must be from the Assistant"
        assert isinstance(assistant_message['content'], list), "This is expected to be a list of parts"
        # The last text part contains the final answer with ####
        last_text_part = assistant_message['content'][-1]['text']
        # Extract both the ground truth answer and the predicted answer
        ref_num = extract_answer(last_text_part)
        pred_num = extract_answer(assistant_response)
        # Compare and return the success as int
        is_correct = int(pred_num == ref_num)
        return is_correct

    def reward(self, conversation, assistant_response):
        """ Use simple 0-1 reward just like gsm8k."""
        is_correct = self.evaluate(conversation, assistant_response)
        is_correct_float = float(is_correct)
        return is_correct_float


class SimpleSpelling(Task):
    """
    SimpleSpelling: Basic word spelling practice.

    A condensed version of SpellingBee focused purely on spelling words
    letter-by-letter, without the counting component. This task isolates
    the core challenge that makes SpellingBee difficult for small models.

    Why this task exists:
    - LLMs see text as tokens (semantic chunks), not character sequences
    - Small models need explicit training to map tokens -> character sequences
    - This provides focused practice on the spelling skill
    - Larger models learn this naturally, but small models benefit from explicit examples

    Example:
        Q: Spell the word: strawberry
        A: strawberry:s,t,r,a,w,b,e,r,r,y

    This is particularly useful for:
    - Midtraining phase (augmenting base knowledge)
    - Small models (< 1B parameters)
    - Tasks requiring character-level understanding
    """

    def __init__(self, size=1000, split="train", **kwargs):
        """
        Initialize the SimpleSpelling task.

        Args:
            size: Number of synthetic spelling problems to generate
            split: Either "train" or "test"
            **kwargs: Additional arguments passed to parent Task
        """
        super().__init__(**kwargs)
        assert split in ["train", "test"], "SpellingBee split must be train|test"
        self.size = size
        self.split = split

        # Load the same word list as SpellingBee
        filename = WORD_LIST_URL.split("/")[-1]
        word_list_path = download_file_with_lock(WORD_LIST_URL, filename)
        with open(word_list_path, 'r', encoding='utf-8') as f:
            words = [line.strip() for line in f]

        # Shuffle the word list with a fixed seed
        # This gives us a different word order than SpellingBee, ensuring variety
        rng = random.Random(42)
        rng.shuffle(words)
        self.words = words

    @property
    def eval_type(self):
        """SimpleSpelling is a generative task."""
        return 'generative'

    def num_examples(self):
        """Returns the configured size (number of problems)."""
        return self.size

    def get_example(self, index):
        """
        Generate a simple spelling problem.

        Args:
            index: Index determining which word to spell

        Returns:
            dict: Conversation with user request and assistant spelling

        Format: "word:l,e,t,t,e,r,s" (word followed by comma-separated characters)
        """
        # Use index as random seed (different seeds for train vs test)
        seed = index if self.split == "train" else -(index + 1)
        rng = random.Random(seed)

        # Pick a random word to spell
        word = rng.choice(self.words)

        # Format as comma-separated letters
        word_letters = ",".join(list(word))

        # Build the simple conversation
        messages = [
            {"role": "user", "content": f"Spell the word: {word}"},
            {"role": "assistant", "content": f"{word}:{word_letters}"}
        ]

        conversation = {
            "messages": messages,
        }
        return conversation


if __name__ == "__main__":

    # preview the SpellingBee task, first 10 examples
    task = SpellingBee()
    for i in range(10):
        ex = task.get_example(i)
        print("=" * 100)
        print(ex['messages'][0]['content'])
        print("-" * 100)
        # Assistant content is now a list of parts
        assistant_parts = ex['messages'][1]['content']
        for part in assistant_parts:
            if part['type'] == 'text':
                print(part['text'], end='')
            elif part['type'] == 'python':
                print(f"<<{part['text']}=", end='')
            elif part['type'] == 'python_output':
                print(f"{part['text']}>>", end='')
        print()
        print("-" * 100)

    # # preview the SimpleSpelling task, first 10 examples
    # task = SimpleSpelling()
    # for i in range(10):
    #     ex = task.get_example(i)
    #     print("=" * 100)
    #     print(ex['messages'][0]['content'])
    #     print("-" * 100)
    #     print(ex['messages'][1]['content'])

    # # also scrutinize the tokenization (last example only)
    # from nanochat.tokenizer import get_tokenizer
    # tokenizer = get_tokenizer()
    # ids, mask = tokenizer.render_conversation(ex)
    # print(tokenizer.visualize_tokenization(ids, mask, with_token_id=True))
