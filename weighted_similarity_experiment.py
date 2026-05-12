import os
from pathlib import Path

import torch
import torch.nn as nn
import pandas as pd
from torchvision.models import resnet18
from safetensors.torch import load_file


# -----------------------------
# 1. Model architecture
# -----------------------------

def make_model():
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(
        3, 64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False
    )
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 100)
    return model


def load_model(checkpoint_path: str, device: str = "cpu"):
    model = make_model()
    state_dict = load_file(checkpoint_path, device=device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


# -----------------------------
# 2. Similarity helpers
# -----------------------------

def flatten_tensor(t: torch.Tensor):
    return t.detach().cpu().float().flatten()


def cosine_similarity(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8):
    a = flatten_tensor(a)
    b = flatten_tensor(b)

    if a.numel() != b.numel():
        return None

    denom = torch.norm(a) * torch.norm(b) + eps
    return torch.dot(a, b).item() / denom.item()


def l2_distance(a: torch.Tensor, b: torch.Tensor):
    a = flatten_tensor(a)
    b = flatten_tensor(b)

    if a.numel() != b.numel():
        return None

    return torch.norm(a - b).item()


def sign_agreement(a: torch.Tensor, b: torch.Tensor):
    a = flatten_tensor(a)
    b = flatten_tensor(b)

    if a.numel() != b.numel():
        return None

    return (torch.sign(a) == torch.sign(b)).float().mean().item()


# -----------------------------
# 3. Compare target vs suspect
# -----------------------------

def compare_models(target_model, suspect_model):
    target_sd = target_model.state_dict()
    suspect_sd = suspect_model.state_dict()

    layer_cosines = []
    layer_l2s = []
    layer_signs = []

    conv_cosines = []
    bn_cosines = []
    fc_cosines = []

    exact_match_count = 0
    total_matchable_layers = 0

    for name, target_tensor in target_sd.items():
        if name not in suspect_sd:
            continue

        suspect_tensor = suspect_sd[name]

        if target_tensor.shape != suspect_tensor.shape:
            continue

        total_matchable_layers += 1

        cos = cosine_similarity(target_tensor, suspect_tensor)
        l2 = l2_distance(target_tensor, suspect_tensor)
        sign = sign_agreement(target_tensor, suspect_tensor)

        if cos is not None:
            layer_cosines.append(cos)
        if l2 is not None:
            layer_l2s.append(l2)
        if sign is not None:
            layer_signs.append(sign)

        if torch.equal(target_tensor.cpu(), suspect_tensor.cpu()):
            exact_match_count += 1

        lower_name = name.lower()

        if "conv" in lower_name and cos is not None:
            conv_cosines.append(cos)

        if "bn" in lower_name and cos is not None:
            bn_cosines.append(cos)

        if "fc" in lower_name and cos is not None:
            fc_cosines.append(cos)

    def safe_mean(values, default=0.0):
        return sum(values) / len(values) if values else default

    features = {
        "mean_cosine": safe_mean(layer_cosines),
        "min_cosine": min(layer_cosines) if layer_cosines else 0.0,
        "max_cosine": max(layer_cosines) if layer_cosines else 0.0,
        "mean_l2": safe_mean(layer_l2s),
        "mean_sign_agreement": safe_mean(layer_signs),
        "conv_cosine": safe_mean(conv_cosines),
        "bn_cosine": safe_mean(bn_cosines),
        "fc_cosine": safe_mean(fc_cosines),
        "exact_layer_fraction": (
            exact_match_count / total_matchable_layers
            if total_matchable_layers > 0 else 0.0
        ),
        "num_matchable_layers": total_matchable_layers,
    }

    return features


# -----------------------------
# 4. Convert features to score
# -----------------------------

def raw_weight_score(features):
    """
    Higher score = more likely stolen.

    This is intentionally conservative:
    - direct copies get very high exact_layer_fraction
    - fine-tuned/checkpoint-derived models get high cosine/sign agreement
    - classifier head and BN stats are useful stealing clues
    """

    # score = (
    #     0.35 * features["mean_cosine"] +
    #     0.20 * features["mean_sign_agreement"] +
    #     0.15 * features["conv_cosine"] +
    #     0.10 * features["bn_cosine"] +
    #     0.10 * features["fc_cosine"] +
    #     0.10 * features["exact_layer_fraction"]
    # )
    score = (
        0.45 * features["conv_cosine"] +
        0.25 * features["bn_cosine"] +
        0.15 * features["mean_sign_agreement"] +
        0.10 * features["mean_cosine"] +
        0.05 * features["exact_layer_fraction"]
    )

    return score


def minmax_normalize(scores):
    scores = torch.tensor(scores, dtype=torch.float32)

    min_score = scores.min().item()
    max_score = scores.max().item()

    if max_score - min_score < 1e-8:
        return [0.5 for _ in scores]

    normalized = (scores - min_score) / (max_score - min_score)
    return normalized.tolist()


# -----------------------------
# 5. Main pipeline
# -----------------------------

def main():
    TARGET_CHECKPOINT = "./target_model/weights.safetensors"
    SUSPECT_DIR = "./suspect_models/"

    OUTPUT_FEATURES = "weight_features.csv"
    OUTPUT_SUBMISSION = "submission.csv"

    target_model = load_model(TARGET_CHECKPOINT)

    rows = []
    raw_scores = []

    for model_id in range(360):
        suspect_path = Path(SUSPECT_DIR) / f"suspect_{model_id:03d}.safetensors"

        if not suspect_path.exists():
            print(f"[WARNING] Missing suspect model: {suspect_path}")
            features = {
                "mean_cosine": 0.0,
                "min_cosine": 0.0,
                "max_cosine": 0.0,
                "mean_l2": 0.0,
                "mean_sign_agreement": 0.0,
                "conv_cosine": 0.0,
                "bn_cosine": 0.0,
                "fc_cosine": 0.0,
                "exact_layer_fraction": 0.0,
                "num_matchable_layers": 0,
            }
            score = 0.0
            raise FileNotFoundError(f"Suspect model not found: {suspect_path}")

        else:
            print(f"Processing suspect model {model_id}: {suspect_path}")

            suspect_model = load_model(str(suspect_path))
            features = compare_models(target_model, suspect_model)
            score = raw_weight_score(features)

        row = {
            "id": model_id,
            **features,
            "raw_score": score,
        }

        rows.append(row)
        raw_scores.append(score)

    normalized_scores = minmax_normalize(raw_scores)

    for row, norm_score in zip(rows, normalized_scores):
        row["score"] = norm_score

    features_df = pd.DataFrame(rows)
    features_df.to_csv(OUTPUT_FEATURES, index=False)

    submission_df = features_df[["id", "score"]]
    submission_df.to_csv(OUTPUT_SUBMISSION, index=False)

    print(f"Saved features to: {OUTPUT_FEATURES}")
    print(f"Saved submission to: {OUTPUT_SUBMISSION}")


if __name__ == "__main__":
    main()