import ast
from .chess_utils.parsing import coerce_response, extract_solution


def compute_score(solution_str, ground_truth_str):
    """
    The scoring function for our chess engine's RL learning loop.
    
    Args:
        solution_str (str): The model's output string.
        ground_truth (str): The ground truth answer (dict of move to reward -- defined in our dataprocessing code)
    """
    ground_truth = ast.literal_eval(ground_truth_str)    # Need to ast literal eval it since stored as a string due to parquet saving issues
    
    # Coerce and catch parse errors
    try:
        predicted_answer = coerce_response(extract_solution(solution_str), task_type="predict_singlemove")
        if len(predicted_answer) > 10:
            predicted_answer = None
    except:
        predicted_answer = None

    # Giving penalty of -0.2 if not legal move
    if predicted_answer is None:
        reward = -0.5
    else:
        reward = ground_truth.get(predicted_answer, -0.2)
    
    # Debugging
    # print(f"Extracted answer: {predicted_answer}; In ground truth: {predicted_answer in ground_truth}; Reward={reward}")
    # print(f"Total Generation:\n{solution_str}\n{'='*60}\n\n\n") 

    return reward