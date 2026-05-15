import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import pandas as pd
from torchvision.models import resnet18
from safetensors.torch import load_file


# ============================================================
# 1. Model architecture: same as task_template.py
# ============================================================

def make_model() -> nn.Module:
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(
        3, 64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 100)
    return model


def load_model(checkpoint_path: str, device: str = "cpu") -> nn.Module:
    model = make_model()
    state_dict = load_file(checkpoint_path, device=device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


# ============================================================
# 2. Tensor similarity helpers
# ============================================================

def flatten(x: torch.Tensor) -> torch.Tensor:
    return x.detach().cpu().float().flatten()


def cosine(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> float:
    a = flatten(a)
    b = flatten(b)

    if a.numel() != b.numel():
        return 0.0

    denom = torch.norm(a) * torch.norm(b) + eps
    return float(torch.dot(a, b) / denom)


def relative_l2_score(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> float:
    """
    Converts relative L2 distance into similarity score.
    Higher = more similar.
    """
    a = flatten(a)
    b = flatten(b)

    if a.numel() != b.numel():
        return 0.0

    rel_l2 = torch.norm(a - b) / (torch.norm(a) + eps)
    return float(1.0 / (1.0 + rel_l2))


def sign_agreement(a: torch.Tensor, b: torch.Tensor) -> float:
    a = flatten(a)
    b = flatten(b)

    if a.numel() != b.numel():
        return 0.0

    return float((torch.sign(a) == torch.sign(b)).float().mean())


def exact_match(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.shape != b.shape:
        return 0.0
    return float(torch.equal(a.cpu(), b.cpu()))


# ============================================================
# 3. Layer group mapping
# ============================================================

def get_group_name(param_name: str) -> str:
    """
    ResNet-18 groups:
    - conv1
    - bn1
    - layer1
    - layer2
    - layer3
    - layer4
    - fc
    - other
    """

    if param_name.startswith("conv1"):
        return "conv1"

    if param_name.startswith("bn1"):
        return "bn1"

    if param_name.startswith("layer1"):
        return "layer1"

    if param_name.startswith("layer2"):
        return "layer2"

    if param_name.startswith("layer3"):
        return "layer3"

    if param_name.startswith("layer4"):
        return "layer4"

    if param_name.startswith("fc"):
        return "fc"

    return "other"


def is_batchnorm_stat(param_name: str) -> bool:
    return (
        "running_mean" in param_name
        or "running_var" in param_name
        or "num_batches_tracked" in param_name
    )


def is_weight_or_bias(param_name: str) -> bool:
    return param_name.endswith(".weight") or param_name.endswith(".bias")


# ============================================================
# 4. Compare target and suspect by layer groups
# ============================================================

def compare_stagewise(target_model: nn.Module, suspect_model: nn.Module) -> Dict[str, float]:
    target_sd = target_model.state_dict()
    suspect_sd = suspect_model.state_dict()

    groups = [
        "conv1",
        "bn1",
        "layer1",
        "layer2",
        "layer3",
        "layer4",
        "fc",
        "other",
    ]

    group_cosines = {g: [] for g in groups}
    group_l2_scores = {g: [] for g in groups}
    group_signs = {g: [] for g in groups}
    group_exact = {g: [] for g in groups}

    all_cosines = []
    all_l2_scores = []
    all_signs = []
    all_exact = []

    bn_stat_cosines = []
    bn_stat_l2_scores = []

    fc_weight_cosines = []
    fc_bias_cosines = []
    fc_weight_l2_scores = []
    fc_bias_l2_scores = []

    missing_count = 0
    shape_mismatch_count = 0
    matched_count = 0

    for name, target_tensor in target_sd.items():
        if name not in suspect_sd:
            missing_count += 1
            continue

        suspect_tensor = suspect_sd[name]

        if target_tensor.shape != suspect_tensor.shape:
            shape_mismatch_count += 1
            continue

        # Skip num_batches_tracked for cosine/sign because it is integer-like and not very informative
        if "num_batches_tracked" in name:
            continue

        matched_count += 1

        g = get_group_name(name)

        cos_val = cosine(target_tensor, suspect_tensor)
        l2_val = relative_l2_score(target_tensor, suspect_tensor)
        sign_val = sign_agreement(target_tensor, suspect_tensor)
        exact_val = exact_match(target_tensor, suspect_tensor)

        group_cosines[g].append(cos_val)
        group_l2_scores[g].append(l2_val)
        group_signs[g].append(sign_val)
        group_exact[g].append(exact_val)

        all_cosines.append(cos_val)
        all_l2_scores.append(l2_val)
        all_signs.append(sign_val)
        all_exact.append(exact_val)

        if is_batchnorm_stat(name):
            bn_stat_cosines.append(cos_val)
            bn_stat_l2_scores.append(l2_val)

        if name == "fc.weight":
            fc_weight_cosines.append(cos_val)
            fc_weight_l2_scores.append(l2_val)

        if name == "fc.bias":
            fc_bias_cosines.append(cos_val)
            fc_bias_l2_scores.append(l2_val)

    def mean(values: List[float]) -> float:
        return float(sum(values) / len(values)) if values else 0.0

    features = {}

    for g in groups:
        features[f"{g}_cosine"] = mean(group_cosines[g])
        features[f"{g}_rel_l2_score"] = mean(group_l2_scores[g])
        features[f"{g}_sign_agreement"] = mean(group_signs[g])
        features[f"{g}_exact_fraction"] = mean(group_exact[g])

    features["mean_cosine"] = mean(all_cosines)
    features["mean_rel_l2_score"] = mean(all_l2_scores)
    features["mean_sign_agreement"] = mean(all_signs)
    features["exact_layer_fraction"] = mean(all_exact)

    features["bn_stat_cosine"] = mean(bn_stat_cosines)
    features["bn_stat_rel_l2_score"] = mean(bn_stat_l2_scores)

    features["fc_weight_cosine"] = mean(fc_weight_cosines)
    features["fc_bias_cosine"] = mean(fc_bias_cosines)
    features["fc_weight_rel_l2_score"] = mean(fc_weight_l2_scores)
    features["fc_bias_rel_l2_score"] = mean(fc_bias_l2_scores)

    features["matched_param_count"] = matched_count
    features["missing_param_count"] = missing_count
    features["shape_mismatch_count"] = shape_mismatch_count

    return features


# ============================================================
# 5. Rank normalization
# ============================================================

def add_rank_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df[f"{col}_rank"] = df[col].rank(method="average", pct=True)
    return df


def normalize_score(df: pd.DataFrame, raw_col: str = "raw_score") -> pd.Series:
    """
    Use rank normalization because leaderboard metric is ranking-based.
    """
    return df[raw_col].rank(method="average", pct=True)


# ============================================================
# 6. Scoring variants for experiments
# ============================================================

def compute_score(df: pd.DataFrame, variant: str) -> pd.DataFrame:
    """
    Each variant produces:
    - raw_score
    - score: rank-normalized [0, 1]
    """

    rank_cols = [
        "conv1_cosine",
        "layer1_cosine",
        "layer2_cosine",
        "layer3_cosine",
        "layer4_cosine",
        "fc_cosine",
        "fc_weight_cosine",
        "fc_bias_cosine",
        "bn1_cosine",
        "bn_stat_cosine",
        "mean_cosine",
        "mean_rel_l2_score",
        "fc_weight_rel_l2_score",
        "layer4_rel_l2_score",
        "exact_layer_fraction",
        "mean_sign_agreement",
    ]

    df = add_rank_columns(df, rank_cols)

    if variant == "v1_late_layers_fc":
        df["raw_score"] = (
            0.05 * df["conv1_cosine_rank"] +
            0.05 * df["layer1_cosine_rank"] +
            0.10 * df["layer2_cosine_rank"] +
            0.15 * df["layer3_cosine_rank"] +
            0.25 * df["layer4_cosine_rank"] +
            0.30 * df["fc_weight_cosine_rank"] +
            0.10 * df["fc_bias_cosine_rank"]
        )

    elif variant == "v2_fc_dominant_stage":
        df["raw_score"] = (
            0.50 * df["fc_weight_cosine_rank"] +
            0.10 * df["fc_bias_cosine_rank"] +
            0.20 * df["layer4_cosine_rank"] +
            0.10 * df["layer3_cosine_rank"] +
            0.10 * df["mean_cosine_rank"]
        )

    elif variant == "v3_backbone_progressive":
        df["raw_score"] = (
            0.05 * df["conv1_cosine_rank"] +
            0.10 * df["layer1_cosine_rank"] +
            0.15 * df["layer2_cosine_rank"] +
            0.25 * df["layer3_cosine_rank"] +
            0.30 * df["layer4_cosine_rank"] +
            0.15 * df["fc_weight_cosine_rank"]
        )

    elif variant == "v4_l2_plus_cosine":
        df["raw_score"] = (
            0.25 * df["fc_weight_cosine_rank"] +
            0.20 * df["layer4_cosine_rank"] +
            0.15 * df["mean_cosine_rank"] +
            0.15 * df["fc_weight_rel_l2_score_rank"] +
            0.15 * df["layer4_rel_l2_score_rank"] +
            0.10 * df["mean_rel_l2_score_rank"]
        )

    elif variant == "v5_bn_sensitive":
        df["raw_score"] = (
            0.30 * df["fc_weight_cosine_rank"] +
            0.20 * df["layer4_cosine_rank"] +
            0.20 * df["bn_stat_cosine_rank"] +
            0.10 * df["bn1_cosine_rank"] +
            0.10 * df["mean_sign_agreement_rank"] +
            0.10 * df["exact_layer_fraction_rank"]
        )

    elif variant == "v6_copy_or_partial_copy":
        """
        Max-signal style:
        if any layer group looks highly copied, rank it high.
        """
        evidence_cols = [
            "conv1_cosine_rank",
            "layer1_cosine_rank",
            "layer2_cosine_rank",
            "layer3_cosine_rank",
            "layer4_cosine_rank",
            "fc_weight_cosine_rank",
            "fc_bias_cosine_rank",
            "exact_layer_fraction_rank",
        ]
        df["raw_score"] = df[evidence_cols].max(axis=1)

    elif variant == "v7_stage_rank_ensemble":
        """
        Ensemble of multiple stage-wise views.
        Recommended after trying v1-v6.
        """
        s_late = (
            0.15 * df["layer3_cosine_rank"] +
            0.30 * df["layer4_cosine_rank"] +
            0.40 * df["fc_weight_cosine_rank"] +
            0.15 * df["fc_bias_cosine_rank"]
        )

        s_backbone = (
            0.05 * df["conv1_cosine_rank"] +
            0.10 * df["layer1_cosine_rank"] +
            0.15 * df["layer2_cosine_rank"] +
            0.30 * df["layer3_cosine_rank"] +
            0.30 * df["layer4_cosine_rank"] +
            0.10 * df["mean_cosine_rank"]
        )

        s_l2 = (
            0.30 * df["fc_weight_rel_l2_score_rank"] +
            0.25 * df["layer4_rel_l2_score_rank"] +
            0.20 * df["mean_rel_l2_score_rank"] +
            0.25 * df["fc_weight_cosine_rank"]
        )

        s_copy = df[
            [
                "fc_weight_cosine_rank",
                "layer4_cosine_rank",
                "exact_layer_fraction_rank",
                "bn_stat_cosine_rank",
            ]
        ].max(axis=1)

        df["raw_score"] = (
            0.35 * s_late +
            0.25 * s_backbone +
            0.20 * s_l2 +
            0.20 * s_copy
        )

    else:
        raise ValueError(f"Unknown variant: {variant}")

    df["score"] = normalize_score(df, "raw_score")
    return df


# ============================================================
# 7. Checkpoint path resolver
# ============================================================

def find_suspect_checkpoint(suspect_dir: Path, model_id: int) -> Path:
    """
    Handles common naming patterns.
    Modify this if your HuggingFace folder uses another format.
    """

    candidates = [
        suspect_dir / f"{model_id}.safetensors",
        suspect_dir / f"model_{model_id}.safetensors",
        suspect_dir / f"suspect_{model_id:03d}.safetensors",
        suspect_dir / str(model_id) / "model.safetensors",
        suspect_dir / str(model_id) / "model.safetensors.index.json",
    ]

    for p in candidates:
        if p.exists() and p.suffix == ".safetensors":
            return p

    # fallback: search recursively for safetensors inside folder named by id
    folder = suspect_dir / str(model_id)
    if folder.exists():
        matches = list(folder.rglob("*.safetensors"))
        if matches:
            return matches[0]

    raise FileNotFoundError(f"No checkpoint found for suspect id {model_id}")


# ============================================================
# 8. Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--target",
        type=str,
        required=True,
        help="Path to target model .safetensors",
    )

    parser.add_argument(
        "--suspect_dir",
        type=str,
        required=True,
        help="Directory containing suspect model checkpoints",
    )

    parser.add_argument(
        "--variant",
        type=str,
        default="v1_late_layers_fc",
        choices=[
            "v1_late_layers_fc",
            "v2_fc_dominant_stage",
            "v3_backbone_progressive",
            "v4_l2_plus_cosine",
            "v5_bn_sensitive",
            "v6_copy_or_partial_copy",
            "v7_stage_rank_ensemble",
        ],
    )

    parser.add_argument(
        "--num_models",
        type=int,
        default=360,
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default="outputs_stagewise",
    )

    args = parser.parse_args()

    target_path = Path(args.target)
    suspect_dir = Path(args.suspect_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading target model: {target_path}")
    target_model = load_model(str(target_path), device="cpu")

    rows = []

    for model_id in range(args.num_models):
        print(f"\nProcessing suspect model {model_id}")

        try:
            suspect_path = find_suspect_checkpoint(suspect_dir, model_id)
            print(f"  checkpoint: {suspect_path}")

            suspect_model = load_model(str(suspect_path), device="cpu")
            features = compare_stagewise(target_model, suspect_model)

            row = {
                "id": model_id,
                "checkpoint_path": str(suspect_path),
                **features,
                "load_error": "",
            }

        except Exception as e:
            print(f"  ERROR: {e}")

            row = {
                "id": model_id,
                "checkpoint_path": "",
                "load_error": str(e),
            }

        rows.append(row)

    df = pd.DataFrame(rows)

    # Fill missing numeric columns with 0
    for col in df.columns:
        if col not in ["checkpoint_path", "load_error"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    features_path = out_dir / "stagewise_weight_features.csv"
    df.to_csv(features_path, index=False)
    print(f"\nSaved features: {features_path}")

    scored_df = compute_score(df.copy(), args.variant)

    scored_path = out_dir / f"stagewise_scored_{args.variant}.csv"
    scored_df.to_csv(scored_path, index=False)

    submission_path = out_dir / f"submission_{args.variant}.csv"
    scored_df[["id", "score"]].to_csv(submission_path, index=False)

    print(f"Saved scored features: {scored_path}")
    print(f"Saved submission: {submission_path}")

    print("\nTop 30 suspects by score:")
    display_cols = [
        "id",
        "score",
        "raw_score",
        "fc_weight_cosine",
        "fc_bias_cosine",
        "layer4_cosine",
        "layer3_cosine",
        "mean_cosine",
        "exact_layer_fraction",
        "bn_stat_cosine",
    ]

    available_cols = [c for c in display_cols if c in scored_df.columns]
    print(
        scored_df.sort_values("score", ascending=False)[available_cols]
        .head(30)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()