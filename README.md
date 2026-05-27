# The Point, the Vision and the Text: Does Point Cloud Boost Spatial Reasoning of Large Language Models?

<div align="center" style="line-height: 1;">
  <a href="https://dl.acm.org/doi/abs/10.1145/3746027.3758219" target="_blank"><img alt="Homepage"
    src="https://img.shields.io/badge/Paper-ACM MM-green.svg"/></a>
  <a href="https://huggingface.co/datasets/EmbodiedCity/Open3DVQA-v2" target="_blank"><img alt="Hugging Face"
    src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Open3DVQA%20-ffc107?color=ffc107&logoColor=white"/></a>
</div>


Despite some promising results, the advantages of point clouds over other modalities remain unclear. Moreover, existing 3D benchmarks are insufficient for fairly evaluating the ability of multimodal LLMs to comprehend spatial concepts. To address these challenges, we introduce ScanReQA, a 3D spatial reasoning benchmark encompassing text, vision, and point cloud modalities.

![Overview](assets/cover.png)

[//]: # (to do: add arxiv link and dataset link)
______________________________________________________________________

## 📢 News
- [x] ScanReQA dataset released
- [x] Evaluation code and cached results released
______________________________________________________________________

## 1. Environment Setup

```bash
conda create -n scanreqa python=3.9 -y
conda activate scanreqa
```

Then install the required packages:
```bash
pip install -r requirements.txt
```

Notes:

- `evaluate/em_acc_recall.py` uses the bundled `meteor-1.5.jar` for the `METEOR` metric. `Java` must be installed on the machine if you want to compute `METEOR`.


<!-- ### 1.3 Data Layout

The repository assumes the following default directories:

- Attention cache:
  - `output_cache/Attention_cache/scanqa_shuffled_token`
  - `output_cache/Attention_cache/relspatialqa_shuffled_token`
  - `output_cache/Attention_cache/scanqa_shuffled_token_full`
  - `output_cache/Attention_cache/relspatialqa_shuffled_token_full`
- Evaluation results:
  - `output_cache/Eval_result_ScanQA/PC`
  - `output_cache/Eval_result_SQA3D/PC`
  - `output_cache/Eval_result_RespatialQA/PC`
  - `output_cache/Eval_result_AbsSpatialQA/PC`
  - `output_cache/Attention_eval_result` -->


## 2. Evaluation
We provide the evaluation code and example outputs for 3D LLMs. To evaluate a custom model, please ensure that its output format is consistent with the provided examples.

Each output file is a json list. Each item in the list corresponds to one evaluation example and has the following fields:

- `source`
  The unique sample identifier. Example: `val-scene0011-0`.
- `scene_id`
  The ScanNet scene identifier for the sample. Example: `scene0011_00`.
- `instruction`
  The full input prompt given to the model, usually including the `USER` question and the `ASSISTANT` prefix.
- `response_gt`
  A list of ground-truth answers. Multiple equivalent reference answers may be provided for the same question.
- `response_pred`
  The model prediction for the sample.

Example:

```json
{
  "source": "val-scene0011-0",
  "scene_id": "scene0011_00",
  "instruction": "USER: What color is the chair in the kitchen? ASSISTANT:",
  "response_gt": ["dark brown", "brown"],
  "response_pred": "brown"
}
```

To run the evaluation code on cached output, download [output_cache](https://huggingface.co/datasets/EmbodiedCity/ScanReQA/tree/main) to the current directory and run the following code.


### 2.1 EM of Different 3D LLMs on ScanQA, SQA3D, and RelSpatialQA
Evaluate different 3D LLMs with `EM` and `refined-EM` on spatial QA benchmarks. Report `CIDEr`, `BLEU-4`, `METEOR`, and `ROUGE` at the same time.


**ScanQA**:

```bash
python evaluate/em_acc_recall.py \
  evaluate-main-results \
  --repo-root . \
  --target-split Eval_result_ScanQA \
  --modal PC \
  --suffix ''
```

**SQA3D**:

```bash
python evaluate/em_acc_recall.py \
  evaluate-main-results \
  --repo-root . \
  --target-split Eval_result_SQA3D \
  --modal PC \
  --suffix ''
```

**RelSpatialQA**:

```bash
python evaluate/em_acc_recall.py \
  evaluate-main-results \
  --repo-root . \
  --target-split Eval_result_RespatialQA \
  --modal PC \
  --suffix ''
```

### 2.2 Accuracy and Recall of 3D LLMs on RelSpatialQA
Measure accuracy and recall on RelSpatialQA.

1. Prepare `Eval_result_RespatialQA/PC` and `Eval_result_ScanQA/PC`.
2. Run the RelSpatialQA command below.

```bash
python evaluate/em_acc_recall.py \
  evaluate-main-results \
  --repo-root . \
  --target-split Eval_result_RespatialQA \
  --scanqa-split Eval_result_ScanQA \
  --modal PC \
  --suffix ''
```

### 2.3 Accuracy of 3D LLMs on AbsSpatialQA
Measure accuracy on AbsSpatialQA.

1. Prepare `Eval_result_AbsSpatialQA/PC`.
2. Run `evaluate-main-results`.

```bash
python evaluate/em_acc_recall.py \
  evaluate-main-results \
  --repo-root . \
  --target-split Eval_result_AbsSpatialQA \
  --modal PC \
  --suffix ''
```

## 3. Attention Visualization

### 3.1 Attention on Tokens
Inspect the sliding-window attention distribution from response tokens to point-cloud tokens. Compare token-level attention patterns between successful and failed cases.

1. Use `output_cache/Attention_cache/{dataset}_shuffled_token`.
2. Use `output_cache/Attention_eval_result/{dataset}.json`.
3. Run `attention-on-tokens`.

**ScanQA**:

```bash
python evaluate/vis_attention.py \
  attention-on-tokens \
  --dataset scanqa
```

RelSpatialQA:

```bash
python evaluate/vis_attention.py \
  attention-on-tokens \
  --dataset relspatialqa
```

Optional arguments:

- `--max-cases`: the maximum evaluated cases
- `--plot`: plot the results

### 3.2 Attention on Target Token
Measure whether the model places attention on the true target-object tokens.

1. Choose the dataset.
2. Run `attention-on-target-token`.

**ScanQA**:

```bash
python evaluate/vis_attention.py \
  attention-on-target-token \
  --dataset scanqa
```

**RelSpatialQA**:

```bash
python evaluate/vis_attention.py \
  attention-on-target-token \
  --dataset relspatialqa
```


### 3.3 Attention on Sink Tokens
Measure how much attention is assigned to sink tokens after shuffling. Compare sink-token attention between successful and failed cases.


**ScanQA**:

```bash
python evaluate/vis_attention.py \
  attention-on-sink-token \
  --dataset scanqa
```

**RelSpatialQA**:

```bash
python evaluate/vis_attention.py \
  attention-on-sink-token \
  --dataset relspatialqa
```


## 4. Logit Lens Evaluation
Inspect the top decoded tokens from point-cloud tokens and response tokens across transformer layers. Analyze when semantic information becomes identifiable during spatial reasoning.


1. Prepare a locally available Vicuna/LLaMA base model.
2. Prepare a full cache directory that contains `all_hidden_states`.
3. Run `evaluate/logit_lens.py`.
4. Read the output from `--save-dir/results.json`.

### 4.1 ScanQA

```bash
python evaluate/logit_lens.py \
  --base-model /path/to/base_model \
  --save-dir output_cache/logit_lens_scanqa \
  --eval-count 1
```

### 4.2 RelSpatialQA

```bash
python evaluate/logit_lens.py \
  --base-model /path/to/base_model \
  --att-dir output_cache/Attention_cache/relspatialqa_shuffled_token_full \
  --save-dir output_cache/logit_lens_relspatialqa \
  --eval-count 1
```
