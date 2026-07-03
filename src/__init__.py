"""Hallucination detection on HaluEval-QA: the notebook champion as a package.

Public API:
    load_qa_data / download_qa_data  - dataset loading (src.data)
    Featurizer / FEATURE_NAMES       - the 13 engineered features (src.features)
    HallucinationScorer / split_by_qid / train_champion - training (src.pipeline)
    evaluate_model / nearest_length_match - raw + length-matched eval (src.evaluate)
"""

from .data import HALUEVAL_QA_URL, download_qa_data, load_qa_data
from .evaluate import evaluate_model, evaluate_predictions, nearest_length_match
from .features import FEATURE_NAMES, Featurizer
from .pipeline import (
    CHAMPION_XGB_PARAMS,
    HallucinationScorer,
    split_by_qid,
    train_champion,
)

__all__ = [
    "CHAMPION_XGB_PARAMS",
    "FEATURE_NAMES",
    "Featurizer",
    "HALUEVAL_QA_URL",
    "HallucinationScorer",
    "download_qa_data",
    "evaluate_model",
    "evaluate_predictions",
    "load_qa_data",
    "nearest_length_match",
    "split_by_qid",
    "train_champion",
]
