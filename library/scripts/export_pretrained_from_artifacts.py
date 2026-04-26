from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

from probemon.core.probe import Probe


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have",
    "if", "in", "is", "it", "of", "on", "or", "that", "the", "this", "to", "with",
}
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
MODEL_ALIASES = ["meta-llama/Meta-Llama-3.1-8B-Instruct"]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def domain_terms(texts: Iterable[str], n: int = 50) -> list[str]:
    counts: Counter[str] = Counter()
    for text in texts:
        words = set(re.findall(r"[a-z0-9][a-z0-9_-]+", text.lower()))
        counts.update(word for word in words if word not in STOPWORDS and len(word) > 2)
    return [word for word, _ in counts.most_common(n)]


def prompt_stats(texts: list[str]) -> dict[str, object]:
    lengths = np.asarray([len(text) for text in texts], dtype=float)
    return {
        "training_prompt_length_mean": float(lengths.mean()),
        "training_prompt_length_std": float(lengths.std(ddof=0)),
        "domain_indicator_terms": domain_terms(texts),
    }


def raw_space_probe(model: LogisticRegression, scaler: StandardScaler) -> tuple[np.ndarray, float]:
    coef = model.coef_.reshape(-1)
    scale = np.where(scaler.scale_ == 0.0, 1.0, scaler.scale_)
    direction = coef / scale
    bias = float(model.intercept_[0] - np.sum(coef * scaler.mean_ / scale))
    return direction.astype(np.float32), bias


def export_legal(root: Path, out_dir: Path) -> None:
    source = np.load(root / "artifacts/runs/residual_correctness_probe_direction.npz", allow_pickle=False)
    eval_payload = json.loads((root / "artifacts/runs/maud_full_probe_proxy_smoke.json").read_text())
    examples = read_jsonl(root / "artifacts/runs/maud_full_examples.jsonl")
    prompts = [f"{row.get('question_text', '')}\n\n{row.get('excerpt_text', '')}" for row in examples]
    metadata = {
        "probe_id": "legal-qa-llama-3.1-8b-v1",
        "model_name": MODEL_NAME,
        "model_aliases": MODEL_ALIASES,
        "layer": 19,
        "pooling": "mean_pool_fixed_maud_claim_token_span",
        "training_dataset": "maud-legal-qa",
        "training_dataset_size": 150,
        "training_label_source": "GPT-5.4 judge-proxy labels",
        "validation_auroc": float(eval_payload["residual"]["correctness"]["auroc"]),
        "test_auroc": 0.7708944281524927,
        "test_brier": float(eval_payload["residual"]["correctness"]["brier"]),
        "version": "0.1.0",
        "domain_description": "MAUD merger-agreement legal QA claims judged by GPT-5.4 proxy labels.",
        "paper_result": "AUROC 0.771 under GPT-5.4 judge; Kimi sensitivity AUROC 0.707.",
        "limitations": [
            "Judge-proxy labels are not expert legal ground truth.",
            "MAUD has only 150 frozen claim units.",
        ],
        **prompt_stats(prompts),
    }
    Probe(
        probe_id=metadata["probe_id"],
        direction=np.asarray(source["residual_direction"], dtype=np.float32),
        bias=float(np.asarray(source["residual_intercept"]).reshape(-1)[0]),
        platt_a=1.0,
        platt_b=0.0,
        metadata=metadata,
    ).save(out_dir / "legal-qa-llama-3.1-8b-v1.npz")


def export_biography(root: Path, out_dir: Path) -> None:
    features = np.load(root / "artifacts/runs/factscore_chatgpt_probe_features_residual.npz", allow_pickle=False)
    x = np.asarray(features["matrix"], dtype=float)
    claim_ids = [str(item) for item in features["claim_ids"].tolist()]
    example_ids = [str(item) for item in features["example_ids"].tolist()]
    labels = {row["claim_id"]: row for row in read_jsonl(root / "data/annotations/factscore_chatgpt_labels.jsonl")}
    examples = {row["example_id"]: row for row in read_jsonl(root / "data/factscore/factscore_chatgpt_examples.jsonl")}
    y = np.asarray([1 if labels[claim_id]["correctness_label"] == "true" else 0 for claim_id in claim_ids], dtype=int)
    splits = np.asarray([examples[example_id]["split"] for example_id in example_ids])
    train_val_idx = np.where(np.isin(splits, ["train", "validation"]))[0]
    test_idx = np.where(splits == "test")[0]
    scaler = StandardScaler()
    x_train_val = scaler.fit_transform(x[train_val_idx])
    model = LogisticRegression(C=0.01, solver="liblinear", max_iter=5000)
    model.fit(x_train_val, y[train_val_idx])
    test_probs = model.predict_proba(scaler.transform(x[test_idx]))[:, 1]
    test_auroc = float(roc_auc_score(y[test_idx], test_probs))
    test_brier = float(brier_score_loss(y[test_idx], test_probs))
    if not (0.78 <= test_auroc <= 0.82):
        raise RuntimeError(f"FActScore probe AUROC {test_auroc:.3f} fell outside expected 0.78-0.82 range")
    direction, bias = raw_space_probe(model, scaler)
    eval_payload = json.loads((root / "artifacts/runs/factscore_chatgpt_validation_eval.json").read_text())
    prompts = [row["question_text"] for row in examples.values()]
    metadata = {
        "probe_id": "biography-llama-3.1-8b-v1",
        "model_name": MODEL_NAME,
        "model_aliases": MODEL_ALIASES,
        "layer": 19,
        "pooling": "parent_sentence_span_mean_pooling_for_canonicalized_atomic_fact_labels",
        "training_dataset": "factscore-biographies",
        "training_dataset_size": int(x.shape[0]),
        "training_label_source": "FActScore human Supported / Not-supported labels; Irrelevant dropped",
        "validation_auroc": float(eval_payload["residual_probe"]["validation"]["selected"]["auroc"]),
        "test_auroc": test_auroc,
        "test_brier": test_brier,
        "version": "0.1.0",
        "domain_description": "FActScore ChatGPT biographies with human atomic-fact labels.",
        "generation_mismatch_caveat": "Biographies were generated by ChatGPT; activations are Llama 3.1 8B reading those biographies.",
        "sentence_span_pooling_caveat": "Atomic facts are canonicalized; activations are pooled over parent sentence spans.",
        "paper_result": "AUROC 0.802 vs Llama self-report 0.541 on 742 held-out atomic facts.",
        **prompt_stats(prompts),
    }
    Probe(
        probe_id=metadata["probe_id"],
        direction=direction,
        bias=bias,
        platt_a=1.0,
        platt_b=0.0,
        metadata=metadata,
    ).save(out_dir / "biography-llama-3.1-8b-v1.npz")


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    out_dir = root / "library/probemon/pretrained/artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    export_legal(root, out_dir)
    export_biography(root, out_dir)
    print(f"Wrote pretrained probes to {out_dir}")


if __name__ == "__main__":
    main()
