# Python file to extract answer from model output and compute reward score
# Direction from this file: https://github.com/volcengine/verl/blob/main/verl/utils/reward_score/gsm8k.py
# Better explained by this: https://verl.readthedocs.io/en/latest/preparation/reward_function.html

# Drop this into the following folder: verl/utils/reward_score/

import re


def extract_solution(text: str) -> str:
    """
    Extracts text between <answer> and </answer> tags, trims it, and returns it.
    The search starts after the third instance of <|im_start|>.
    Returns None if no valid extraction is found.

    Args:
        text (str): The input string containing the <answer> tags.

    Returns:
        str: The trimmed extracted text or None if extraction fails.
    """
    failure_response = None

    # Find the position of the third occurrence of <|im_start|>
    im_start_positions = [m.start() for m in re.finditer(r"<\|im_start\|>", text)]

    if len(im_start_positions) < 3:
        raise ValueError("Insufficient occurrences of '<|im_start|>' in text.")

    subtext_start = im_start_positions[2] + len("<|im_start|>")  # Start after the third occurrence
    subtext = text[subtext_start:]

    # Extract text within <answer>...</answer> in the subtext
    matches = re.findall(r"<answer>(.*?)</answer>", subtext, re.DOTALL)

    if not matches:
        return failure_response  # No match found

    extracted = matches[-1].strip()  # Take the last match

    return extracted if re.fullmatch(r"[A-Za-z0-9+\-#]{1,6}", extracted) else failure_response



def compute_score(solution_str, ground_truth, method='strict', format_score=0.1):
    """
    The scoring function for our chess engine's RL learning loop.
    
    Args:
        solution_str (str): The model's output string.
        ground_truth (dict): The ground truth answer (dict of move to reward -- defined in our dataprocessing code)
    """
    answer = extract_solution(solution_str)
    # print(f"Extracted answer: {answer}; In ground truth: {answer in ground_truth}")  # Debugging line
    # print(f"Total Generation:\n{solution_str}\n{'='*60}\n\n\n") 

    if answer is None:
        # Extraction error -- not formatted properly
        return -0.1
    else:
        if answer in ground_truth:
            return ground_truth[answer] + format_score
        else:
            # Optionally give a small reward for at least giving us a valid looking move
            return format_score