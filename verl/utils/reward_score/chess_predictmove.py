import ast
from .chess_utils.parsing import coerce_response, extract_solution


def compute_score(solution_str, ground_truth_str, format_score=0.0):
    """
    The scoring function for our chess engine's RL learning loop.
    
    Args:
        solution_str (str): The model's output string.
        ground_truth (str): The ground truth answer (dict of move to reward -- defined in our dataprocessing code)
    """
    ground_truth = ast.literal_eval(ground_truth_str)    # Need to ast literal eval it since stored as a string due to parquet saving issues
    try:
        predicted_answer = coerce_response(extract_solution(solution_str), task_type="predict_singlemove")
    except:
        # Catch for any extraction / coercion errors
        predicted_answer = None

    # Debugging
    if True:
        print(f"Extracted answer: {predicted_answer}; In ground truth: {predicted_answer in ground_truth}")
        print(f"Total Generation:\n{solution_str}\n{'='*60}\n\n\n") 

    # Return reward
    if predicted_answer is None:
        return -0.1
    else:
        if predicted_answer in ground_truth:
            return ground_truth[predicted_answer] + format_score
        else:
            return format_score