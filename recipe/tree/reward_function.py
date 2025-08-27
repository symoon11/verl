from typing import Any

from math_verify.metric import math_metric
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig


def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info: dict[str, Any] = None) -> float:
    if "</think>" in solution_str:
        solution_str = solution_str.split("</think>")[-1]
        ground_truth = f"\\boxed{{{ground_truth}}}"
        verify_func = math_metric(
            gold_extraction_target=(LatexExtractionConfig(),),
            pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
        )
        score, _ = verify_func([ground_truth], [solution_str])
    else:
        score = 0.0
    return score
