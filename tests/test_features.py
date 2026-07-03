import math

import numpy as np
import pandas as pd
import pytest

from src.features import (
    FEATURE_NAMES,
    Featurizer,
    all_tokens,
    content_tokens,
    ground_overlap,
    is_substr,
    lcs_char_ratio,
    lcs_token_ratio,
    split_sentences,
)


def make_fitted(texts=("apple banana", "apple cherry")):
    return Featurizer().fit(list(texts))


class TestTokenizers:
    def test_content_tokens_drop_stopwords_and_case(self):
        assert content_tokens("The Great Wall of China") == ["great", "wall", "china"]

    def test_all_tokens_keep_stopwords(self):
        assert all_tokens("The red Fox!") == ["the", "red", "fox"]

    def test_split_sentences(self):
        assert split_sentences("A b. C d! E?") == ["A b.", "C d!", "E?"]
        assert split_sentences("no terminator") == ["no terminator"]


class TestGroundOverlap:
    def test_exact_fraction(self):
        assert ground_overlap("Paris Texas", "Paris is the capital of France") == 0.5

    def test_full_overlap(self):
        assert ground_overlap("Paris", "Paris is the capital of France") == 1.0

    def test_empty_answer_is_zero(self):
        assert ground_overlap("", "some knowledge") == 0.0

    def test_stopword_only_answer_is_zero(self):
        assert ground_overlap("the of an", "the of an") == 0.0


class TestIsSubstr:
    def test_positive_with_trailing_period_and_case(self):
        assert is_substr("The Eiffel Tower.", "the eiffel tower is in Paris") == 1.0

    def test_negative(self):
        assert is_substr("Berlin", "the eiffel tower is in Paris") == 0.0

    def test_empty_after_normalization(self):
        assert is_substr("  .", "anything") == 0.0


class TestLcsRatios:
    def test_char_ratio_full_match(self):
        assert lcs_char_ratio("abc", "zzabczz") == 1.0

    def test_char_ratio_partial(self):
        assert lcs_char_ratio("abcd", "ab--cd") == 0.5

    def test_char_ratio_answer_longer_than_knowledge(self):
        assert lcs_char_ratio("a very long answer indeed", "long") == pytest.approx(4 / 25)

    def test_char_ratio_empty_answer(self):
        assert lcs_char_ratio("", "knowledge") == 0.0

    def test_token_ratio_contiguous_run(self):
        # common contiguous token run is ["red", "fox"]: 2 of 3 answer tokens
        assert lcs_token_ratio("the red fox", "a red fox ran") == pytest.approx(2 / 3)

    def test_token_ratio_no_common_tokens(self):
        assert lcs_token_ratio("alpha beta", "gamma delta") == 0.0


class TestComputeFeatures:
    def test_requires_fit(self):
        with pytest.raises(RuntimeError):
            Featurizer().compute_features("q", "a", "k")

    def test_returns_all_13_features(self):
        feats = make_fitted().compute_features("Who?", "apple", "apple pie")
        assert list(feats) == FEATURE_NAMES

    def test_numeric_features(self):
        feats = make_fitted().compute_features(
            "When was he born?",
            "He was born in 1985 and had 3 dogs",
            "Born in 1985.",
        )
        assert feats["num_frac_in_know"] == 0.5  # 1985 found, 3 missing
        assert feats["n_num_missing"] == 1.0
        assert feats["year_mismatch"] == 0.0

    def test_year_mismatch(self):
        feats = make_fitted().compute_features(
            "When?", "Released in 1999 and 2005", "It came out in 1999"
        )
        assert feats["year_mismatch"] == 1.0

    def test_sentence_concentration_and_spread(self):
        know = "John lives in Paris. Mary lives in Rome."
        one_sentence = make_fitted().compute_features("Who?", "John Paris", know)
        assert one_sentence["sent_overlap_max"] == 1.0
        assert one_sentence["overlap_spread"] == 0.0
        # stitched across two sentences: whole-passage overlap 1.0, best sentence 0.5
        stitched = make_fitted().compute_features("Who?", "John Rome", know)
        assert stitched["sent_overlap_max"] == 0.5
        assert stitched["overlap_spread"] == 0.5

    def test_novel_overlap_ignores_question_tokens(self):
        feats = make_fitted().compute_features(
            "Who is John Smith?", "John Smith", "Bob was here"
        )
        assert feats["novel_overlap"] == 1.0  # nothing novel beyond the question
        feats = make_fitted().compute_features(
            "Who wrote it?", "Mark Twain", "nothing relevant here"
        )
        assert feats["novel_overlap"] == 0.0

    def test_idf_overlap_known_corpus(self):
        featurizer = make_fitted()  # apple in both docs -> idf 0; banana in one
        grounded = featurizer.compute_features("q", "apple banana", "banana tree")
        assert grounded["idf_overlap"] == pytest.approx(1.0)
        zero_den = featurizer.compute_features("q", "apple", "banana")
        assert zero_den["idf_overlap"] == 0.0  # only zero-idf tokens -> denominator 0

    def test_negation_features(self):
        mismatch = make_fitted().compute_features(
            "Did he go?", "He did not go", "He went to the store."
        )
        assert mismatch["ans_has_neg"] == 1.0
        assert mismatch["neg_mismatch"] == 1.0
        agree = make_fitted().compute_features(
            "Did he go?", "He did not go", "He did not go to the store."
        )
        assert agree["neg_mismatch"] == 0.0

    def test_contraction_counts_as_negation(self):
        feats = make_fitted().compute_features("Did he?", "He didn't go", "He went.")
        assert feats["ans_has_neg"] == 1.0

    def test_empty_strings_produce_finite_defaults(self):
        feats = make_fitted().compute_features("", "", "")
        assert all(math.isfinite(v) for v in feats.values())
        assert feats["num_frac_in_know"] == 1.0
        assert feats["novel_overlap"] == 1.0
        assert feats["ground_overlap"] == 0.0

    def test_unicode_does_not_crash(self):
        feats = make_fitted().compute_features("Qui?", "Café Münchén ★", "Le Café est ici")
        assert all(math.isfinite(v) for v in feats.values())


class TestTransform:
    def test_transform_shape_and_columns(self):
        frame = pd.DataFrame(
            {
                "question": ["Who?", "Who?"],
                "answer": ["apple", "unrelated"],
                "knowledge": ["apple pie", "apple pie"],
            }
        )
        matrix = make_fitted().transform(frame)
        assert list(matrix.columns) == FEATURE_NAMES
        assert matrix.shape == (2, 13)
        assert matrix.dtypes.eq(np.float32).all()

    def test_roundtrip_serialization(self):
        featurizer = make_fitted()
        clone = Featurizer.from_dict(featurizer.to_dict())
        assert clone.ndoc == featurizer.ndoc
        assert clone.compute_features("q", "apple banana", "banana tree") == (
            featurizer.compute_features("q", "apple banana", "banana tree")
        )
