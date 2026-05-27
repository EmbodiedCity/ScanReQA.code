import argparse
import importlib
import json
import os
import sys
from typing import Dict, Optional

import yaml
from box import Box

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from evaluate.egc_eval import answer_match, clean_answer


def _prepend_repo_root(repo_root: str):
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def _get_metric_scorers(code_root: str):
    _prepend_repo_root(code_root)
    bleu_cls = importlib.import_module("evaluate.evaluator.ngram_metrics.bleu.bleu").Bleu
    cider_cls = importlib.import_module("evaluate.evaluator.ngram_metrics.cider.cider").Cider
    meteor_cls = importlib.import_module("evaluate.evaluator.ngram_metrics.meteor.meteor").Meteor
    rouge_cls = importlib.import_module("evaluate.evaluator.ngram_metrics.rouge.rouge").Rouge
    return cider_cls(), bleu_cls(), meteor_cls(), rouge_cls()


def _load_eval_cfg(code_root: str):
    candidate_paths = [
        os.path.join(code_root, "config", "eval.yaml"),
        os.path.join(code_root, "output_cache", "eval.yaml"),
    ]
    for config_path in candidate_paths:
        if os.path.exists(config_path):
            with open(config_path, "r") as handle:
                return Box(yaml.safe_load(handle))
    raise FileNotFoundError(f"No evaluation config found under {code_root}")


def _resolve_eval_dir(repo_root: str, split: str, modal: str, suffix: str) -> str:
    preferred = os.path.join(repo_root, "output_cache", split, f"{modal}{suffix}")
    if os.path.isdir(preferred):
        return preferred

    split_root = os.path.join(repo_root, "output_cache", split)
    if not os.path.isdir(split_root):
        raise FileNotFoundError(split_root)

    candidates = [
        os.path.join(split_root, name)
        for name in sorted(os.listdir(split_root))
        if os.path.isdir(os.path.join(split_root, name))
    ]
    if len(candidates) == 1:
        return candidates[0]
    raise FileNotFoundError(preferred)


def _eval_from_result_legacy(code_root: str, result_path: str):
    cider_scorer, bleu_scorer, meteor_scorer, rouge_scorer = _get_metric_scorers(code_root)

    with open(result_path, "r") as handle:
        results = json.load(handle)

    total_count = len(results)
    print(total_count)

    em = 0
    em_refined = 0
    success_qas = []
    pred_sentence_mp = []
    gt_sentence_mp = []

    for record in results:
        answer_pred = clean_answer(record["response_pred"])
        answer_gts = [clean_answer(gt) for gt in record["response_gt"]]
        em_flag, em_refined_flag = answer_match(answer_pred, answer_gts)
        em += em_flag
        em_refined += em_refined_flag
        if em_refined_flag:
            success_qas.append(record)
        pred_sentence_mp.append([answer_pred])
        gt_sentence_mp.append(answer_gts)

    print(len(success_qas))

    gt_sentence_mp = {idx: value for idx, value in enumerate(gt_sentence_mp)}
    pred_sentence_mp = {idx: value for idx, value in enumerate(pred_sentence_mp)}

    eval_dict = {
        "target_metric": em_refined / total_count,
        "em": em / total_count,
        "em_refined": em_refined / total_count,
        "cider": cider_scorer.compute_score(gt_sentence_mp, pred_sentence_mp)[0],
        "bleu": bleu_scorer.compute_score(gt_sentence_mp, pred_sentence_mp)[0][-1],
        "meteor": meteor_scorer.compute_score(gt_sentence_mp, pred_sentence_mp)[0],
        "rouge": rouge_scorer.compute_score(gt_sentence_mp, pred_sentence_mp)[0],
    }
    return (True, eval_dict), success_qas


def _eval_abs_spatial_acc(data_path: str):
    with open(data_path, "r") as handle:
        results = json.load(handle)

    succ_count = 0.0
    total_count = 0.0
    for record in results:
        pred = record["response_pred"].lower()
        gt = record["response_gt"][0].lower()
        if pred == gt:
            succ_count += 1
        total_count += 1

    print(f"abs succ rate: {succ_count}/ {total_count} = {succ_count / total_count}")


def _eval_acc_recall(data_file_path1: str, data_file_path2: str):
    with open(data_file_path1, "r") as handle:
        output_data = json.load(handle)

    good_cases = []
    bad_cases = []
    for record in output_data:
        answer_pred = clean_answer(record["response_pred"])
        answer_gts = [clean_answer(gt) for gt in record["response_gt"]]
        _, refined_em_flag = answer_match(answer_pred, answer_gts)
        if refined_em_flag:
            good_cases.append(record)
        else:
            bad_cases.append(record)

    good_case_id1 = list(set(res["source"] for res in good_cases))
    bad_case_id1 = list(set(res["source"] for res in bad_cases))
    case_id1 = list(set(good_case_id1 + bad_case_id1))
    print(f"good case 1: {len(good_case_id1)}, bad case 1:{len(bad_case_id1)}, total: {len(case_id1)}")

    with open(data_file_path2, "r") as handle:
        output_data = json.load(handle)

    good_cases2 = []
    bad_cases2 = []
    for record in output_data:
        answer_pred = clean_answer(record["response_pred"])
        answer_gts = [clean_answer(gt) for gt in record["response_gt"]]
        _, refined_em_flag = answer_match(answer_pred, answer_gts)
        if refined_em_flag:
            good_cases2.append(record)
        else:
            bad_cases2.append(record)

    good_case_id2 = list(set(res["source"][:-2] for res in good_cases2))
    bad_case_id2 = list(set(res["source"][:-2] for res in bad_cases2))
    case_id2 = list(set(good_case_id2 + bad_case_id2))
    print(f"good case 2: {len(good_case_id2)}, bad case 2:{len(bad_case_id2)}, total: {len(case_id2)}")

    gg = []
    gb = []
    bg = []
    bb = []

    for qid in good_case_id1:
        if qid not in case_id2:
            continue
        if qid in bad_case_id2:
            gb.append(qid)
        else:
            gg.append(qid)

    for qid in bad_case_id1:
        if qid not in case_id2:
            continue
        if qid in bad_case_id2:
            bb.append(qid)
        else:
            bg.append(qid)

    print(f"gg/(gg+gb) = {len(gg)}/ {(len(gg) + len(gb))} = {len(gg) * 1.0 / (len(gg) + len(gb))}")
    print(f"bg/(bg+bb) = {len(bg)}/{len(bg) + len(bb)} = {len(bg) * 1.0 / (len(bg) + len(bb))}")

    good_case_count = 0.0
    for qid in good_case_id2:
        if qid not in bad_case_id2:
            good_case_count += 1

    print(f"acc: {good_case_count} / {len(case_id2)} = {good_case_count / len(case_id2)}")
    print(f"recall = {len(gg)}/ {(len(gg) + len(gb))} = {len(gg) * 1.0 / (len(gg) + len(gb))}")
    print(len(gg + gb))
    print(gg + gb)
    print(len(gb))
    print(gb)


def evaluate_main_results(
    repo_root: str,
    modal: str = "Text",
    target_split: str = "Eval_result_SQA3D",
    scanqa_split: str = "Eval_result_ScanQA",
    suffix: str = "_updated_model",
):
    code_root = repo_root
    _load_eval_cfg(code_root)
    root = _resolve_eval_dir(repo_root, target_split, modal, suffix)
    scanqa_root = _resolve_eval_dir(repo_root, scanqa_split, modal, suffix)

    summary: Dict[str, Dict] = {}
    for filename in sorted(os.listdir(root)):
        if not filename.endswith(".json"):
            continue
        result_path = os.path.join(root, filename)
        print("=" * 60)
        print(result_path)
        result, _ = _eval_from_result_legacy(code_root, result_path)
        summary[filename] = result
        print(result)

        if target_split == "Eval_result_AbsSpatialQA":
            _eval_abs_spatial_acc(result_path)

        if target_split in {"Eval_result_RelSpatialQA", "Eval_result_RespatialQA"}:
            scanqa_res_path = os.path.join(scanqa_root, filename)
            _eval_acc_recall(scanqa_res_path, result_path)
    return summary


def _parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    eval_main = subparsers.add_parser("evaluate-main-results")
    eval_main.add_argument("--repo-root", required=True)
    eval_main.add_argument("--modal", default="Text")
    eval_main.add_argument("--target-split", default="Eval_result_SQA3D")
    eval_main.add_argument("--scanqa-split", default="Eval_result_ScanQA")
    eval_main.add_argument("--suffix", default="")

    return parser.parse_args()


def main():
    args = _parse_args()
    if args.command == "evaluate-main-results":
        evaluate_main_results(
            repo_root=args.repo_root,
            modal=args.modal,
            target_split=args.target_split,
            scanqa_split=args.scanqa_split,
            suffix=args.suffix,
        )
        return
    raise SystemExit(
        "Usage:\n"
        "  python evaluate/em_acc_recall.py evaluate-main-results --repo-root /path/to/repo_root"
    )


if __name__ == "__main__":
    main()
