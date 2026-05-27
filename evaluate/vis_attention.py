import argparse
import os
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from evaluate.egc_eval import instance_eval


DATASET_CHOICES = ["relspatialqa", "scanqa"]
TASK_OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "output_cache")
ATTENTION_CACHE_ROOT = os.path.join(TASK_OUTPUT_ROOT, "Attention_cache")
EVAL_RESULT_ROOTS = [
    os.path.join(TASK_OUTPUT_ROOT, "Attention_eval_result"),
    os.path.join(TASK_OUTPUT_ROOT, "Eval_result_shuffled_token"),
]


def visualize_1d_matrix(matrix1, matrix2, caption="", label1="mat1", label2="mat2"):
    x = np.arange(len(matrix1))
    plt.figure(figsize=(10, 5))
    plt.plot(x, matrix1, label=label1, color="blue", linewidth=2)
    plt.plot(x, matrix2, label=label2, color="red", linewidth=2)
    plt.xlabel("Layer")
    plt.ylabel("Attention Ratio")
    plt.title(caption)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def visualize_multi_1d_matrix(matrix, caption=""):
    x = np.arange(matrix.shape[1])
    plt.figure(figsize=(10, 5))
    for index in range(matrix.shape[0]):
        plt.plot(x, matrix[index], label=f"mat_{index}", linewidth=2)
    plt.xlabel("Token Index")
    plt.ylabel("Value")
    plt.title(caption)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def visualize_token_weights(matrix, **kwargs):
    rows, cols = matrix.shape
    fig, axes = plt.subplots(rows, 1, figsize=(10, 3 * rows), sharex=True)
    if rows == 1:
        axes = [axes]

    x = np.arange(cols)
    for idx in range(rows):
        ax = axes[idx]
        ax.plot(x, matrix[idx], linewidth=1.5)
        ax.set_title(f"Layer {idx + 1}", fontsize=12)
        ax.set_ylabel("Weight")
        ax.grid(True)

    axes[-1].set_xlabel("Token Index")
    plt.title(f"{kwargs['qid']}: {kwargs['question']}")
    plt.tight_layout()
    plt.show()


def compute_attention_scores_A2B_v2(att_weight, A_start, A_end, B_start, B_end, split=False):
    n_layer, n_token, _ = att_weight.shape
    A_end = n_token if A_end == -1 else A_end
    B_end = n_token if B_end == -1 else B_end
    return att_weight[:, A_start:A_end, B_start:B_end]


def _base_qid_from_att_file(att_file: str) -> str:
    raw_qid = os.path.splitext(att_file)[0]
    return raw_qid.rsplit("-", 1)[0]


def _resolve_dataset_paths(
    dataset: str,
    att_dir: Optional[str] = None,
    result_path: Optional[str] = None,
) -> Tuple[str, str]:
    if att_dir is None:
        att_dir = os.path.join(ATTENTION_CACHE_ROOT, f"{dataset}_shuffled_token")
    if result_path is None:
        for eval_root in EVAL_RESULT_ROOTS:
            candidate = os.path.join(eval_root, f"{dataset}.json")
            if os.path.isfile(candidate):
                result_path = candidate
                break
        if result_path is None:
            result_path = os.path.join(EVAL_RESULT_ROOTS[0], f"{dataset}.json")
    if not os.path.isdir(att_dir):
        raise FileNotFoundError(f"Attention cache directory not found: {att_dir}")
    if not os.path.isfile(result_path):
        raise FileNotFoundError(f"Evaluation result file not found: {result_path}")
    return att_dir, result_path


def _mean_head_attention(attention_map: torch.Tensor) -> np.ndarray:
    return attention_map.cpu().numpy().mean(axis=2)


def _iter_attention_cases(att_dir: str) -> Iterable[Dict]:
    att_files = sorted(os.listdir(att_dir))
    for att_file in tqdm(att_files):
        att_map_dict = torch.load(os.path.join(att_dir, att_file), map_location="cpu")
        batch_attention = _mean_head_attention(att_map_dict["all_attentions"])
        batch_token_idxes = att_map_dict["token_idxes"].cpu().numpy()
        batch_obj_ids = (
            att_map_dict["obj_ids"].cpu().numpy()
            if att_map_dict["obj_ids"] is not None
            else [np.arange(0, len(att_map_dict["selected_obj_ids"][i])) for i in range(len(att_map_dict["qid"]))]
        )
        batch_shuffled_idxes = (
            att_map_dict["shuffled_idxes"].cpu().numpy()
            if "shuffled_idxes" in att_map_dict and att_map_dict["shuffled_idxes"] is not None
            else None
        )

        for index, qid in enumerate(att_map_dict["qid"]):
            yield {
                "qid": qid,
                "att_file": att_file,
                "question": att_map_dict["question"][index],
                "att_weight": batch_attention[:, index, :, :],
                "token_idxes": batch_token_idxes[index],
                "obj_ids": batch_obj_ids[index],
                "shuffled_idxes": None if batch_shuffled_idxes is None else batch_shuffled_idxes[index],
            }


def _resp_to_pc_attention(att_weight: np.ndarray, token_idxes: Sequence[int]) -> np.ndarray:
    pc_start = int(token_idxes[-3])
    pc_end = int(token_idxes[-2])
    resp_start = int(token_idxes[-1])
    resp_end = -1
    scores = compute_attention_scores_A2B_v2(att_weight, resp_start, resp_end, pc_start, pc_end, split=True)
    scores = np.stack(scores, axis=0)
    scores = scores.sum(axis=1)
    return scores[:, 1:]


def _reindex_by_shuffle(att_scores: np.ndarray, shuffled_idxes: Sequence[int]) -> np.ndarray:
    reordered = att_scores.copy()
    target_positions = [int(np.where(i == shuffled_idxes)[0][0]) for i in range(len(shuffled_idxes))]
    reordered[:, list(range(len(shuffled_idxes)))] = att_scores[:, target_positions]
    return reordered


def _success_qids(result_path: str) -> set:
    _, success_cases = instance_eval(result_path)
    return {case["source"] for case in success_cases}


def _plot_if_needed(succ: np.ndarray, fail: np.ndarray, caption: str, plot: bool) -> None:
    if plot:
        visualize_1d_matrix(succ, fail, caption=caption, label1="succ", label2="fail")


def _safe_mean_stack(values: List[np.ndarray], width: int = 32) -> np.ndarray:
    if not values:
        return np.zeros(width, dtype=np.float32)
    return np.stack(values, axis=0).mean(axis=0)


def compute_attention_on_sink_token(
    dataset: str,
    att_dir: Optional[str] = None,
    result_path: Optional[str] = None,
    max_cases: int = 10000,
    plot: bool = False,
):
    att_dir, result_path = _resolve_dataset_paths(dataset, att_dir, result_path)
    success_qids = _success_qids(result_path)
    att_rats_succ: List[np.ndarray] = []
    att_rats_fail: List[np.ndarray] = []
    att_rats: List[np.ndarray] = []

    for case in _iter_attention_cases(att_dir):
        att_scores = _resp_to_pc_attention(case["att_weight"], case["token_idxes"])
        target_obj_len = len(case["obj_ids"])
        target_att_score = att_scores[:, :target_obj_len].mean(axis=1)
        avg_att_score = att_scores.mean(axis=1)
        att_ratio = target_att_score / avg_att_score

        if case["qid"] in success_qids:
            att_rats_succ.append(att_ratio)
        else:
            att_rats_fail.append(att_ratio)
        att_rats.append(att_ratio)

        if len(att_rats) >= max_cases:
            break

    att_rats_succ = _safe_mean_stack(att_rats_succ)
    att_rats_fail = _safe_mean_stack(att_rats_fail)
    all_att_rats = np.stack(att_rats, axis=0).mean(axis=0)

    print("\t".join(map(str, att_rats_succ.tolist())))
    print("\t".join(map(str, att_rats_fail.tolist())))
    _plot_if_needed(att_rats_succ, att_rats_fail, "relspatialqa_target_attention", plot)
    return {
        "success": att_rats_succ,
        "failure": att_rats_fail,
        "all": all_att_rats,
    }


def compute_attention_on_target_token(
    dataset: str,
    max_cases: int,
    att_dir: Optional[str] = None,
    result_path: Optional[str] = None,
    plot: bool = False,
):
    att_dir, result_path = _resolve_dataset_paths(dataset, att_dir, result_path)
    if dataset == "relspatialqa":
        with open(result_path, "r") as handle:
            total_qas = [qa["source"] for qa in __import__("json").load(handle)]
        _, success_cases = instance_eval(result_path)
        success_qas = [qa["source"] for qa in success_cases]
        success_qid_set = set(success_qas)
        fail_qas = list(set(total_qas) - set(success_qas))

        success_groups = set(qid[:-2] for qid in success_qas)
        failure_groups = set(qid[:-2] for qid in fail_qas)
        full_success = success_groups - failure_groups
        full_failure = failure_groups - success_groups

        att_rats_succ: List[np.ndarray] = []
        att_rats_fail: List[np.ndarray] = []
        target_base_qids = full_success | full_failure
        att_files = sorted(
            att_file
            for att_file in os.listdir(att_dir)
            if _base_qid_from_att_file(att_file) in target_base_qids
        )
        fallback_to_case_level = False
        if not att_files:
            fallback_to_case_level = True
            att_files = sorted(os.listdir(att_dir))
            print("No complete good/bad groups matched the available attention dumps. Falling back to case-level success/failure.")

        for att_file in tqdm(att_files):
            att_map_dict = torch.load(os.path.join(att_dir, att_file), map_location="cpu")
            batch_attention = _mean_head_attention(att_map_dict["all_attentions"])
            batch_token_idxes = att_map_dict["token_idxes"].cpu().numpy()
            batch_obj_ids = att_map_dict["obj_ids"].cpu().numpy()
            batch_shuffled_idxes = att_map_dict["shuffled_idxes"].cpu().numpy()

            for index, qid in enumerate(att_map_dict["qid"]):
                base_qid = qid[:-2]
                if not fallback_to_case_level and base_qid not in target_base_qids:
                    continue

                att_scores = _resp_to_pc_attention(batch_attention[:, index, :, :], batch_token_idxes[index])
                att_scores = _reindex_by_shuffle(att_scores, batch_shuffled_idxes[index])
                target_obj_len = len(batch_obj_ids[index])
                target_att_score = att_scores[:, :target_obj_len].mean(axis=1)
                avg_att_score = att_scores.mean(axis=1)
                att_ratio = target_att_score / avg_att_score

                if fallback_to_case_level:
                    if qid in success_qid_set:
                        att_rats_succ.append(att_ratio)
                    else:
                        att_rats_fail.append(att_ratio)
                elif base_qid in full_success:
                    att_rats_succ.append(att_ratio)
                else:
                    att_rats_fail.append(att_ratio)

                if len(att_rats_succ) + len(att_rats_fail) >= max_cases:
                    break
            if len(att_rats_succ) + len(att_rats_fail) >= max_cases:
                break

        att_rats_succ = _safe_mean_stack(att_rats_succ)
        att_rats_fail = _safe_mean_stack(att_rats_fail)

        _plot_if_needed(att_rats_succ, att_rats_fail, "attention on target tokens: good vs bad cases", plot)
        print(att_rats_succ.sum())
        print(att_rats_fail.sum())
        return {"success": att_rats_succ, "failure": att_rats_fail}

    if dataset == "scanqa":
        success_qids = _success_qids(result_path)
        att_rats_succ: List[np.ndarray] = []
        att_rats_fail: List[np.ndarray] = []
        total_att_rats: List[np.ndarray] = []
        sink_att_rats: List[np.ndarray] = []
        tgt_att_rats: List[np.ndarray] = []

        for case in _iter_attention_cases(att_dir):
            att_scores = _resp_to_pc_attention(case["att_weight"], case["token_idxes"])
            target_obj_len = len(case["obj_ids"])
            avg_att_score = att_scores.mean(axis=1).reshape(-1, 1)

            total_att_ratio = att_scores / avg_att_score
            sink_att_ratio = att_scores[:, :target_obj_len] / avg_att_score

            reindexed_scores = _reindex_by_shuffle(att_scores, case["shuffled_idxes"])
            target_att_ratio = reindexed_scores[:, :target_obj_len] / avg_att_score

            sink_att_rats.append(sink_att_ratio.sum(-1))
            tgt_att_rats.append(target_att_ratio.sum(-1))
            total_att_rats.append(total_att_ratio.sum(-1))

            if case["qid"] in success_qids:
                att_rats_succ.append(target_att_ratio.mean(-1))
            else:
                att_rats_fail.append(target_att_ratio.mean(-1))

            if len(att_rats_succ) + len(att_rats_fail) >= max_cases:
                break

        att_rats_succ = _safe_mean_stack(att_rats_succ)
        att_rats_fail = _safe_mean_stack(att_rats_fail)
        total_att_rats = np.stack(total_att_rats, axis=0).mean(axis=0)
        sink_att_rats = np.stack(sink_att_rats, axis=0).mean(axis=0)
        tgt_att_rats = np.stack(tgt_att_rats, axis=0).mean(axis=0)

        _plot_if_needed(att_rats_succ, att_rats_fail, "scanqa_target_attention_shuffle_fix", plot)
        print("\t".join(map(str, total_att_rats.tolist())))
        print("\t".join(map(str, sink_att_rats.tolist())))
        print("\t".join(map(str, tgt_att_rats.tolist())))
        print(att_rats_succ.sum())
        print(att_rats_fail.sum())

        return {
            "success": att_rats_succ,
            "failure": att_rats_fail,
            "total": total_att_rats,
            "sink": sink_att_rats,
            "target": tgt_att_rats,
        }

    raise ValueError(f"Unsupported dataset: {dataset}")


def compute_attention_on_tokens(
    dataset: str,
    max_cases: int = 200,
    att_dir: Optional[str] = None,
    result_path: Optional[str] = None,
    plot: bool = False,
):
    att_dir, result_path = _resolve_dataset_paths(dataset, att_dir, result_path)
    _, succ_qas = instance_eval(result_path)
    succ_qas = [qa["source"] for qa in succ_qas]
    att_files = os.listdir(att_dir)

    att_rats_succ = []
    att_rats_fail = []
    total_slide_att_rats = []
    for file_index in tqdm(range(len(att_files))):
        att_file = att_files[file_index]
        file_path = os.path.join(att_dir, att_file)
        att_map_dict = torch.load(file_path, map_location="cpu")
        batch_attention = att_map_dict["all_attentions"].numpy()
        batch_attention = batch_attention.mean(axis=2)

        batch_selected_obj_ids = att_map_dict["selected_obj_ids"].cpu().numpy()
        batch_token_idxes = att_map_dict["token_idxes"].cpu().numpy()
        batch_question = att_map_dict["question"]
        batch_obj_ids = att_map_dict["obj_ids"].cpu().numpy()
        batch_qid = att_map_dict["qid"]
        batch_shuffled_idxes = att_map_dict["shuffled_idxes"].cpu().numpy()

        batch_size = len(batch_qid)
        for batch_index in range(batch_size):
            att_weight = batch_attention[:, batch_index, :, :]
            token_idxes = batch_token_idxes[batch_index]
            question = batch_question[batch_index]
            obj_ids = batch_obj_ids[batch_index]
            qid = batch_qid[batch_index]
            shuffled_idxes = batch_shuffled_idxes[batch_index]
            reindex = [[], []]
            target_obj_len = len(obj_ids)

            for obj_index in range(len(shuffled_idxes)):
                reindex[0].append(obj_index)
                reindex[1].append(np.where(obj_index == shuffled_idxes)[0][0])

            pc_start = int(token_idxes[-3])
            pc_end = int(token_idxes[-2])
            resp_start = int(token_idxes[-1])
            resp_end = -1
            att_score_per_layer = compute_attention_scores_A2B_v2(
                att_weight,
                resp_start,
                resp_end,
                pc_start,
                pc_end,
                split=True,
            )
            att_score_per_layer = np.stack(att_score_per_layer, axis=0)
            att_score_per_layer = att_score_per_layer.sum(axis=1)
            att_score_per_layer = att_score_per_layer[:, 1:]

            avg_att_score = att_score_per_layer.mean(axis=1)

            slide_att_rats = []
            padded_att_score = np.concatenate(
                [att_score_per_layer, np.zeros((att_score_per_layer.shape[0], target_obj_len - 1))],
                axis=1,
            )
            for slide_index in range(att_score_per_layer.shape[1]):
                slide_att_score = padded_att_score[:, slide_index:slide_index + len(obj_ids)]
                slide_att_rat = slide_att_score.mean(-1) / avg_att_score
                slide_att_rats.append(slide_att_rat.sum())

            att_score_per_layer_reindex = att_score_per_layer.copy()
            att_score_per_layer[:, reindex[0]] = att_score_per_layer_reindex[:, reindex[1]]

            target_att_score = att_score_per_layer[:, :target_obj_len]
            tgt_att_rat = target_att_score.mean(-1) / avg_att_score
            slide_att_rats.append(tgt_att_rat.sum())

            total_slide_att_rats.append(slide_att_rats)

            if qid in succ_qas:
                att_rats_succ.append(tgt_att_rat.mean(-1))
            else:
                att_rats_fail.append(tgt_att_rat.mean(-1))

        if len(att_rats_succ) + len(att_rats_fail) > max_cases:
            break

    att_rats_succ = np.stack(att_rats_succ, axis=0).mean(axis=0)
    att_rats_fail = np.stack(att_rats_fail, axis=0).mean(axis=0)
    total_slide_att_rats = np.stack(total_slide_att_rats, axis=0).mean(0)

    if plot:
        visualize_multi_1d_matrix(total_slide_att_rats.reshape(1, -1), caption="slide_att_rats")

    print("\t".join(map(str, total_slide_att_rats.tolist())))
    print(att_rats_succ.sum())
    print(att_rats_fail.sum())

    return {
        "success": att_rats_succ,
        "failure": att_rats_fail,
        "slide": total_slide_att_rats,
    }


def _parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    attention_on_sink_token = subparsers.add_parser("attention-on-sink-token")
    attention_on_sink_token.add_argument("--dataset", required=True, choices=DATASET_CHOICES)
    attention_on_sink_token.add_argument("--att-dir")
    attention_on_sink_token.add_argument("--result-path")
    attention_on_sink_token.add_argument("--max-cases", type=int, default=10000)
    attention_on_sink_token.add_argument("--plot", action="store_true")

    attention_on_target_token = subparsers.add_parser("attention-on-target-token")
    attention_on_target_token.add_argument("--dataset", required=True, choices=DATASET_CHOICES)
    attention_on_target_token.add_argument("--att-dir")
    attention_on_target_token.add_argument("--result-path")
    attention_on_target_token.add_argument("--max-cases", type=int, default=10000)
    attention_on_target_token.add_argument("--plot", action="store_true")

    attention_on_tokens = subparsers.add_parser("attention-on-tokens")
    attention_on_tokens.add_argument("--dataset", required=True, choices=DATASET_CHOICES)
    attention_on_tokens.add_argument("--att-dir")
    attention_on_tokens.add_argument("--result-path")
    attention_on_tokens.add_argument("--max-cases", type=int, default=10000)
    attention_on_tokens.add_argument("--plot", action="store_true")

    return parser.parse_args()


def main():
    args = _parse_args()
    if args.command == "attention-on-sink-token":
        compute_attention_on_sink_token(
            dataset=args.dataset,
            att_dir=args.att_dir,
            result_path=args.result_path,
            max_cases=args.max_cases,
            plot=args.plot,
        )
        return
    if args.command == "attention-on-target-token":
        compute_attention_on_target_token(
            dataset=args.dataset,
            att_dir=args.att_dir,
            result_path=args.result_path,
            max_cases=args.max_cases,
            plot=args.plot,
        )
        return
    if args.command == "attention-on-tokens":
        compute_attention_on_tokens(
            dataset=args.dataset,
            att_dir=args.att_dir,
            result_path=args.result_path,
            max_cases=args.max_cases,
            plot=args.plot,
        )
        return
    raise SystemExit(
        "Usage:\n"
        "  python evaluate/vis_attention.py attention-on-sink-token --dataset relspatialqa|scanqa\n"
        "  python evaluate/vis_attention.py attention-on-target-token --dataset relspatialqa|scanqa\n"
        "  python evaluate/vis_attention.py attention-on-tokens --dataset relspatialqa|scanqa"
    )


if __name__ == "__main__":
    main()
