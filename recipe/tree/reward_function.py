from typing import Any

from verl.utils.reward_score import math_verify


def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info: dict[str, any] = None) -> float:
    if "</think>" in solution_str:
        model_output = solution_str.split("</think>")[-1]
        score = math_verify.compute_score(model_output, ground_truth)
    else:
        score = 0.0
    return score
