import argparse
import json
import os
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(PROJECT_ROOT) == "evaluate":
    PROJECT_ROOT = os.path.dirname(PROJECT_ROOT)
DEFAULT_ATT_DIR = os.path.join(PROJECT_ROOT, "output_cache", "Attention_cache", "scanqa_shuffled_token_full")


def apply_rms_norm(hidden_states: torch.Tensor, norm_weight: torch.Tensor, eps: float) -> torch.Tensor:
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    return norm_weight.to(hidden_states.dtype) * hidden_states


def logit_len_for_leo(hidden_states, norm_weight, norm_eps, lm_head_weight, tokenizer, device, token_idx=None):
    hidden_states = hidden_states.squeeze(1) if hidden_states.ndim == 4 else hidden_states.squeeze()
    if token_idx is not None:
        hidden_states = hidden_states[:, token_idx[0]:token_idx[1], :]

    hidden_states = hidden_states.to(device)
    norm_weight = norm_weight.to(device)
    lm_head_weight = lm_head_weight.to(device)
    num_layers = len(hidden_states)
    sequence_length = hidden_states[0].size(0)
    all_top_tokens = []

    for layer in range(num_layers):
        layer_hidden_states = hidden_states[layer]
        normalized = apply_rms_norm(layer_hidden_states, norm_weight, norm_eps)
        if normalized.dtype != lm_head_weight.dtype:
            normalized = normalized.to(lm_head_weight.dtype)
        logits = F.linear(normalized, lm_head_weight)
        probs = torch.softmax(logits, dim=-1)
        _, top_5_indices = torch.topk(probs, k=5, dim=-1)

        decoded_token = []
        for pos in range(sequence_length):
            top_5_tokens = [tokenizer.decode(idx.item()) for idx in top_5_indices[pos]]
            decoded_token.append(top_5_tokens[0])
        print(decoded_token)
        all_top_tokens.append(decoded_token)

    return all_top_tokens


def eval_logit_len(norm_weight, norm_eps, lm_head_weight, tokenizer, device, data_root, save_dir, eval_count=-1, selected_qids=None):
    if selected_qids is not None:
        state_data_files = [f"{qid}.pth" for qid in selected_qids]
        for filename in state_data_files:
            if not os.path.exists(os.path.join(data_root, filename)):
                raise FileNotFoundError(f"Could not find state file: {filename} under {data_root}")
    else:
        state_data_files = sorted(os.listdir(data_root))
    if eval_count == -1:
        eval_count = len(state_data_files)

    results = {}
    for index, filename in enumerate(state_data_files):
        if index >= eval_count:
            break

        state_dict = torch.load(os.path.join(data_root, filename), map_location="cpu")
        hidden_states = state_dict["all_hidden_states"]
        token_idxes = state_dict["token_idxes"][0]
        item_id = state_dict["qid"][0]
        question = state_dict["question"][0]
        shuffled_idxes = state_dict["shuffled_idxes"][0]
        obj_ids = state_dict["obj_ids"][0]

        print(item_id)
        print(question)

        pc_token_idx = (token_idxes[-3], token_idxes[-2])
        resp_token_idx = (token_idxes[-1], hidden_states.shape[2])
        pc_top_tokens = logit_len_for_leo(
            hidden_states,
            norm_weight,
            norm_eps,
            lm_head_weight,
            tokenizer,
            device,
            token_idx=pc_token_idx,
        )
        resp_top_tokens = logit_len_for_leo(
            hidden_states,
            norm_weight,
            norm_eps,
            lm_head_weight,
            tokenizer,
            device,
            token_idx=resp_token_idx,
        )

        all_top_tokens = [pc_top_tokens[layer] + ["###"] + resp_top_tokens[layer] for layer in range(len(pc_top_tokens))]
        results[item_id] = {
            "question": question,
            "obj_ids": obj_ids.cpu().tolist(),
            "shuffled_idxes": shuffled_idxes.cpu().tolist(),
            "token_idxes": token_idxes.cpu().tolist(),
            "logit_lens": all_top_tokens,
            "output_gt": state_dict["output_gt"][0],
            "output_pred": state_dict["output_pred"][0],
            "best_sequence_pred": state_dict["best_sequence_pred"][0].cpu().tolist(),
            "beam_output_text": state_dict["beam_output_text"],
            "beam_sequences_scores": state_dict["beam_sequences_scores"][0].cpu().tolist(),
            "beam_sequences": state_dict["beam_sequences"][0].cpu().tolist(),
        }

    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "results.json"), "w") as handle:
        json.dump(results, handle, indent=4)
    return results


def _resolve_shard_paths(base_model: str):
    index_path = os.path.join(base_model, "pytorch_model.bin.index.json")
    with open(index_path, "r") as handle:
        index_data = json.load(handle)
    weight_map = index_data["weight_map"]
    norm_shard = os.path.join(base_model, weight_map["model.norm.weight"])
    lm_head_shard = os.path.join(base_model, weight_map["lm_head.weight"])
    return norm_shard, lm_head_shard


def _load_projection_from_model(base_model: str):
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        local_files_only=True,
        trust_remote_code=False,
        low_cpu_mem_usage=True,
    )
    try:
        norm_weight = model.model.norm.weight.detach().cpu()
    except AttributeError:
        norm_weight = model.model.model.norm.weight.detach().cpu()
    lm_head_weight = model.lm_head.weight.detach().cpu()
    del model
    return norm_weight, lm_head_weight


def build_projection_modules(
    base_model: str,
    device: Optional[str] = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        truncation_side="right",
        local_files_only=True,
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    config = AutoConfig.from_pretrained(base_model, local_files_only=True)
    norm_weight = None
    lm_head_weight = None

    index_path = os.path.join(base_model, "pytorch_model.bin.index.json")
    if os.path.exists(index_path):
        norm_shard, lm_head_shard = _resolve_shard_paths(base_model)
        if os.path.exists(norm_shard) and os.path.exists(lm_head_shard):
            norm_state = torch.load(norm_shard, map_location="cpu")
            if norm_shard == lm_head_shard:
                lm_head_state = norm_state
            else:
                lm_head_state = torch.load(lm_head_shard, map_location="cpu")

            norm_weight = norm_state["model.norm.weight"]
            lm_head_weight = lm_head_state["lm_head.weight"]

    if norm_weight is None or lm_head_weight is None:
        try:
            norm_weight, lm_head_weight = _load_projection_from_model(base_model)
        except Exception as exc:
            raise FileNotFoundError(
                "Could not load model projection weights from the provided base model. "
                f"Checked sharded files under {base_model} and standard transformers loading failed."
            ) from exc

    if device == "cpu":
        norm_weight = norm_weight.float()
        lm_head_weight = lm_head_weight.float()
    return tokenizer, norm_weight, config.rms_norm_eps, lm_head_weight, device


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--att-dir", default=DEFAULT_ATT_DIR)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--eval-count", type=int, default=1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--selected-qids", nargs="*", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.path.isdir(args.att_dir):
        raise FileNotFoundError(f"Attention cache directory not found: {args.att_dir}")
    tokenizer, norm_weight, norm_eps, lm_head_weight, device = build_projection_modules(args.base_model, args.device)
    print(f"Running on {device}")
    return eval_logit_len(
        norm_weight=norm_weight,
        norm_eps=norm_eps,
        lm_head_weight=lm_head_weight,
        tokenizer=tokenizer,
        device=device,
        data_root=args.att_dir,
        save_dir=args.save_dir,
        eval_count=args.eval_count,
        selected_qids=args.selected_qids,
    )


if __name__ == "__main__":
    main()
