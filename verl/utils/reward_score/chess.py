import ast
from .chess_utils.parsing import coerce_response, extract_solution, pqt_extract_ground_truth


DATASOURCE_MAPPING = {
    "chess_predictmove": "predict_singlemove",
    "chess_bestmove": "choose_from_n",
    "chess_worstmove": "choose_from_n",
    "chess_legalmoves": "produce_list"
}


def compute_score(solution_str, ground_truth_str, datasource, verbose=True):
    """
    The scoring function for our chess engine's RL learning loop.
    
    Args:
        solution_str (str): The model's output string.
        ground_truth (str): The ground truth answer (dict of move to reward -- defined in our dataprocessing code)
        datasource   (str): Key relating to the datasource sample is from
    """
    # Start by getting our task type for score computation
    assert datasource in DATASOURCE_MAPPING
    task_type = DATASOURCE_MAPPING[datasource]

    try: 
        ground_truth = pqt_extract_ground_truth(ground_truth_str, task_type)
        predicted_answer = coerce_response(extract_solution(solution_str, task_type))
        if len(predicted_answer) > 100:
            predicted_answer = None
    except Exception as e:
        print(f"Exception encountered: {e}")
        predicted_answer = None

    reward = 0

    # Tasks like 'bestmove' or 'worstmove'
    if task_type == "choose_from_n":
        gt_ans, gt_candidates = ground_truth['answer'], ground_truth['candidates']

        if verbose:
            print(f"[{task_type}] Extracted answer: {predicted_answer}; In candidates? {predicted_answer in gt_candidates}; Correct? {predicted_answer == gt_ans}")
        
        if predicted_answer == gt_ans:
            reward += 1
        elif predicted_answer not in gt_candidates:
            reward -= 0.2

    # Tasks like 'legalmoves'
    elif task_type == "produce_list":
        tp, tp_fp_fn = _score_legalmoves(predicted_answer, ground_truth)
        tp_reward = tp / tp_fp_fn    # If div by 0 then error w/ underlying data; should never happen.

        if verbose:
            print(f"[{task_type}] Extracted answer: {predicted_answer}; TP: {tp}; TP+FP+FN: {tp_fp_fn}; Reward = {tp_reward}")
        reward += tp_reward

    # Tasks like 'predictmove'
    elif task_type == "predict_singlemove":
        if verbose:
            print(f"Extracted answer: {predicted_answer}; In ground truth? {predicted_answer in ground_truth}; Reward = {reward}")
        if predicted_answer is None:
            reward += -0.5
        else:
            reward += ground_truth.get(predicted_answer, -0.2)

    # Debugging
    # print(f"Total Generation:\n{solution_str}\n{'='*60}\n\n\n") 

    return reward




# =========================================
# Helper Funcs
# =========================================
def _score_legalmoves(predicted, ground_truth):
    tp, tp_fp_fn = 0, 0
    guessed = set()

    if isinstance(predicted, list):
        for move in predicted:
            tp_fp_fn += 1
            if move in ground_truth and move not in guessed:
                tp += 1
                guessed.add(move)
            
    tp_fp_fn += (len(ground_truth) - tp)   # Have to include fp

    return tp, tp_fp_fn