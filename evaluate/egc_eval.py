import json
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


def clean_answer(text: str) -> str:
    text = text.lower().strip(".")
    text = re.sub(r"[ ]+$", "", text)
    text = re.sub(r"^[ ]+", "", text)
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\.[ ]{2,}", ". ", text)
    text = re.sub(r"[^a-zA-Z0-9,'\s\-:]+", "", text)
    text = re.sub(r"ç", "c", text)
    text = re.sub(r"’", "'", text)
    text = re.sub(r"\bletf\b", "left", text)
    text = re.sub(r"\blet\b", "left", text)
    text = re.sub(r"\btehre\b", "there", text)
    text = re.sub(r"\brigth\b", "right", text)
    text = re.sub(r"\brght\b", "right", text)
    text = re.sub(r"\bbehine\b", "behind", text)
    text = re.sub(r"\btv\b", "TV", text)
    text = re.sub(r"\bchai\b", "chair", text)
    text = re.sub(r"\bwasing\b", "washing", text)
    text = re.sub(r"\bwaslked\b", "walked", text)
    text = re.sub(r"\boclock\b", "o'clock", text)
    text = re.sub(r"\bo'[ ]+clock\b", "o'clock", text)
    text = re.sub(r"\b0\b", "zero", text)
    text = re.sub(r"\bnone\b", "zero", text)
    text = re.sub(r"\b1\b", "one", text)
    text = re.sub(r"\b2\b", "two", text)
    text = re.sub(r"\b3\b", "three", text)
    text = re.sub(r"\b4\b", "four", text)
    text = re.sub(r"\b5\b", "five", text)
    text = re.sub(r"\b6\b", "six", text)
    text = re.sub(r"\b7\b", "seven", text)
    text = re.sub(r"\b8\b", "eight", text)
    text = re.sub(r"\b9\b", "nine", text)
    text = re.sub(r"\b10\b", "ten", text)
    text = re.sub(r"\b11\b", "eleven", text)
    text = re.sub(r"\b12\b", "twelve", text)
    text = re.sub(r"\b13\b", "thirteen", text)
    text = re.sub(r"\b14\b", "fourteen", text)
    text = re.sub(r"\b15\b", "fifteen", text)
    text = re.sub(r"\b16\b", "sixteen", text)
    text = re.sub(r"\b17\b", "seventeen", text)
    text = re.sub(r"\b18\b", "eighteen", text)
    text = re.sub(r"\b19\b", "nineteen", text)
    text = re.sub(r"\b20\b", "twenty", text)
    text = re.sub(r"\b23\b", "twenty-three", text)
    text = re.sub(r"\b([a-zA-Z]+)([0-9])\b", r"\g<1>", text)
    text = re.sub(r"\ba\b ([a-zA-Z]+)", r"\g<1>", text)
    text = re.sub(r"\ban\b ([a-zA-Z]+)", r"\g<1>", text)
    text = re.sub(r"\bthe\b ([a-zA-Z]+)", r"\g<1>", text)
    text = re.sub(r"\bbackwards\b", "backward", text)
    return text


def answer_match(pred: str, gts: Sequence[str]) -> Tuple[int, int]:
    for gt in gts:
        if pred == gt:
            return 1, 1
        if "".join(pred.split()) in "".join(gt.split()):
            return 0, 1
        if "".join(gt.split()) in "".join(pred.split()):
            return 0, 1
    return 0, 0


def evaluate_results(records: Sequence[Dict]) -> Tuple[Dict[str, float], List[Dict]]:
    total = len(records)
    em = 0
    em_refined = 0
    success_cases: List[Dict] = []

    for record in records:
        pred = clean_answer(record["response_pred"])
        gts = [clean_answer(gt) for gt in record["response_gt"]]
        em_flag, em_refined_flag = answer_match(pred, gts)
        em += em_flag
        em_refined += em_refined_flag
        if em_refined_flag:
            success_cases.append(record)

    metrics = {
        "total_count": total,
        "em": em / total if total else 0.0,
        "em_refined": em_refined / total if total else 0.0,
        "target_metric": em_refined / total if total else 0.0,
    }
    return metrics, success_cases


def evaluate_results_file(path: str) -> Tuple[Dict[str, float], List[Dict]]:
    with open(path, "r") as handle:
        records = json.load(handle)
    return evaluate_results(records)


def instance_eval(path: str) -> Tuple[Dict[str, float], List[Dict]]:
    return evaluate_results_file(path)


def load_json(path: str):
    with open(path, "r") as handle:
        return json.load(handle)


def dump_json(path: str, data) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(data, handle, indent=4)
