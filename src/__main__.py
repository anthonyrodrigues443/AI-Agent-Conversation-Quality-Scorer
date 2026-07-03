"""CLI: python -m src {download,train,evaluate,score}.

Examples (from the repo root):
    python -m src download
    python -m src train --data data/raw/qa_data.json --model-dir models/eng_xgboost
    python -m src evaluate --data data/raw/qa_data.json --model-dir models/eng_xgboost
    python -m src score --model-dir models/eng_xgboost \
        --question "Who wrote Hamlet?" --answer "Shakespeare" \
        --knowledge "Hamlet is a tragedy written by William Shakespeare."
"""

import argparse
import json
from pathlib import Path

from .data import DEFAULT_RAW_PATH, HALUEVAL_QA_URL, download_qa_data, load_qa_data
from .evaluate import evaluate_model
from .pipeline import DEFAULT_QID_FILE, HallucinationScorer, train_champion


def _add_data_args(parser):
    parser.add_argument("--data", default=str(DEFAULT_RAW_PATH), help="path to qa_data.json")
    parser.add_argument(
        "--qid-file",
        default=str(DEFAULT_QID_FILE),
        help="frozen split file (falls back to a fresh grouped split if missing)",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m src", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_download = sub.add_parser("download", help="download HaluEval-QA qa_data.json")
    p_download.add_argument("--dest", default=str(DEFAULT_RAW_PATH))
    p_download.add_argument("--url", default=HALUEVAL_QA_URL)

    p_train = sub.add_parser("train", help="train the champion and report raw+matched metrics")
    _add_data_args(p_train)
    p_train.add_argument("--model-dir", default="models/eng_xgboost")

    p_eval = sub.add_parser("evaluate", help="evaluate a saved model on the held-out split")
    _add_data_args(p_eval)
    p_eval.add_argument("--model-dir", default="models/eng_xgboost")

    p_score = sub.add_parser("score", help="score one question/answer/knowledge triple")
    p_score.add_argument("--model-dir", default="models/eng_xgboost")
    p_score.add_argument("--question", required=True)
    p_score.add_argument("--answer", required=True)
    p_score.add_argument("--knowledge", required=True)

    args = parser.parse_args(argv)

    if args.command == "download":
        dest = download_qa_data(dest=args.dest, url=args.url)
        print(f"data ready at {dest}")
        return

    if args.command == "score":
        scorer = HallucinationScorer.load(args.model_dir)
        p = scorer.score(args.question, args.answer, args.knowledge)
        verdict = "hallucinated" if p >= scorer.threshold else "grounded"
        print(json.dumps({"p_hallucinated": round(p, 4), "verdict": verdict}))
        return

    frame = load_qa_data(args.data)
    qid_file = args.qid_file if Path(args.qid_file).exists() else None
    if qid_file is None:
        print(f"(qid file {args.qid_file} not found: using a fresh grouped split, seed 42)")

    if args.command == "train":
        scorer, train, test = train_champion(frame, qid_file=qid_file)
        scorer.save(args.model_dir)
        print(f"trained on {len(train)} rows, saved to {args.model_dir}")
        report = evaluate_model(scorer, test)
    else:  # evaluate
        from .pipeline import split_by_qid

        scorer = HallucinationScorer.load(args.model_dir)
        _, test = split_by_qid(frame, qid_file=qid_file)
        report = evaluate_model(scorer, test)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
