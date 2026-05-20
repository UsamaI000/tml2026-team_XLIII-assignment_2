import argparse
import json
import random
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from torchvision import datasets, transforms
from torchvision.models import resnet18
from safetensors.torch import load_file


# ============================================================
# 1. Reproducibility
# ============================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# 2. Model architecture from task_template.py
# ============================================================

def make_model() -> nn.Module:
    model = resnet18(weights=None)

    # CIFAR-style ResNet-18
    model.conv1 = nn.Conv2d(
        3,
        64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )

    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 100)

    return model


def load_model(checkpoint_path: str, device: str) -> nn.Module:
    model = make_model()
    state_dict = load_file(checkpoint_path, device="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


# ============================================================
# 3. Assignment-specific normalization and augmentation
# ============================================================

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


class BiasedRandomCrop:
    """
    Assignment-style biased random crop.

    Target training details:
    - reflection padding = 4
    - bias_x = 0.5
    - bias_y = -0.25
    - jitter = 0.25

    CIFAR image:
    - original size: 32x32
    - after padding 4: 40x40
    - crop back to 32x32
    """

    def __init__(
        self,
        size: int = 32,
        padding: int = 4,
        bias_x: float = 0.5,
        bias_y: float = -0.25,
        jitter: float = 0.25,
    ):
        self.size = size
        self.padding = padding
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.jitter = jitter
        self.reflect_pad = transforms.Pad(
            padding=self.padding,
            padding_mode="reflect",
        )

    def __call__(self, img):
        img = self.reflect_pad(img)

        width, height = img.size
        max_x = width - self.size
        max_y = height - self.size

        # Biased center location
        center_x = max_x / 2.0 + self.bias_x * (max_x / 2.0)
        center_y = max_y / 2.0 + self.bias_y * (max_y / 2.0)

        # Random jitter around biased center
        jitter_x = random.uniform(-self.jitter, self.jitter) * max_x
        jitter_y = random.uniform(-self.jitter, self.jitter) * max_y

        left = int(round(center_x + jitter_x))
        top = int(round(center_y + jitter_y))

        left = max(0, min(left, max_x))
        top = max(0, min(top, max_y))

        return img.crop((left, top, left + self.size, top + self.size))


def get_transform(transform_type: str):
    if transform_type == "clean":
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ])

    if transform_type == "target_aug":
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            BiasedRandomCrop(
                size=32,
                padding=4,
                bias_x=0.5,
                bias_y=-0.25,
                jitter=0.25,
            ),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ])

    raise ValueError(f"Unknown transform type: {transform_type}")


# ============================================================
# 4. Dataset loading using train_main_idx.json
# ============================================================

def load_train_main_indices(train_idx_path: str) -> List[int]:
    with open(train_idx_path, "r") as f:
        indices = json.load(f)

    return [int(i) for i in indices]


def build_dataset(
    data_root: str,
    split: str,
    transform_type: str,
    train_idx_path: Optional[str],
    max_samples: Optional[int],
    seed: int,
):
    transform = get_transform(transform_type)

    if split == "test":
        dataset = datasets.CIFAR100(
            root=data_root,
            train=False,
            download=True,
            transform=transform,
        )

    else:
        train_dataset = datasets.CIFAR100(
            root=data_root,
            train=True,
            download=True,
            transform=transform,
        )

        if train_idx_path is None:
            raise ValueError("train_idx_path is required for train-based splits.")

        target_indices = load_train_main_indices(train_idx_path)
        target_set = set(target_indices)

        if split == "train_target":
            dataset = Subset(train_dataset, target_indices)

        elif split == "train_non_target":
            all_indices = set(range(len(train_dataset)))
            non_target_indices = sorted(list(all_indices - target_set))
            dataset = Subset(train_dataset, non_target_indices)

        elif split == "train_all":
            dataset = train_dataset

        else:
            raise ValueError(f"Unknown split: {split}")

    if max_samples is not None and max_samples > 0 and max_samples < len(dataset):
        rng = np.random.default_rng(seed)
        selected = rng.choice(len(dataset), size=max_samples, replace=False)
        dataset = Subset(dataset, selected.tolist())

    return dataset


def build_loader(
    data_root: str,
    split: str,
    transform_type: str,
    train_idx_path: Optional[str],
    batch_size: int,
    num_workers: int,
    max_samples: Optional[int],
    seed: int,
):
    dataset = build_dataset(
        data_root=data_root,
        split=split,
        transform_type=transform_type,
        train_idx_path=train_idx_path,
        max_samples=max_samples,
        seed=seed,
    )

    print(f"Probe dataset size: {len(dataset)}")


    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return loader


# ============================================================
# 5. Functional similarity metrics
# ============================================================

def batch_logit_cosine(target_logits, suspect_logits):
    return F.cosine_similarity(target_logits, suspect_logits, dim=1)


def batch_prob_cosine(target_logits, suspect_logits):
    target_probs = F.softmax(target_logits, dim=1)
    suspect_probs = F.softmax(suspect_logits, dim=1)
    return F.cosine_similarity(target_probs, suspect_probs, dim=1)


def batch_inverse_kl(target_logits, suspect_logits, temperature: float = 2.0):
    """
    KL(target || suspect), converted to similarity.

    Higher score means more similar.
    """
    target_probs = F.softmax(target_logits / temperature, dim=1)
    suspect_log_probs = F.log_softmax(suspect_logits / temperature, dim=1)

    kl = F.kl_div(
        suspect_log_probs,
        target_probs,
        reduction="none",
    ).sum(dim=1)

    return 1.0 / (1.0 + kl)


def batch_symmetric_inverse_kl(target_logits, suspect_logits, temperature: float = 2.0):
    target_probs = F.softmax(target_logits / temperature, dim=1)
    suspect_probs = F.softmax(suspect_logits / temperature, dim=1)

    target_log_probs = F.log_softmax(target_logits / temperature, dim=1)
    suspect_log_probs = F.log_softmax(suspect_logits / temperature, dim=1)

    kl_target_to_suspect = F.kl_div(
        suspect_log_probs,
        target_probs,
        reduction="none",
    ).sum(dim=1)

    kl_suspect_to_target = F.kl_div(
        target_log_probs,
        suspect_probs,
        reduction="none",
    ).sum(dim=1)

    sym_kl = 0.5 * (kl_target_to_suspect + kl_suspect_to_target)

    return 1.0 / (1.0 + sym_kl)


def batch_top1_agreement(target_logits, suspect_logits):
    target_pred = target_logits.argmax(dim=1)
    suspect_pred = suspect_logits.argmax(dim=1)
    return (target_pred == suspect_pred).float()


def batch_top5_overlap(target_logits, suspect_logits):
    target_top5 = target_logits.topk(5, dim=1).indices
    suspect_top5 = suspect_logits.topk(5, dim=1).indices

    overlaps = []

    for i in range(target_logits.shape[0]):
        t = set(target_top5[i].detach().cpu().tolist())
        s = set(suspect_top5[i].detach().cpu().tolist())
        overlaps.append(len(t.intersection(s)) / 5.0)

    return torch.tensor(overlaps, device=target_logits.device)


def batch_margin_similarity(target_logits, suspect_logits):
    target_top2 = target_logits.topk(2, dim=1).values
    suspect_top2 = suspect_logits.topk(2, dim=1).values

    target_margin = target_top2[:, 0] - target_top2[:, 1]
    suspect_margin = suspect_top2[:, 0] - suspect_top2[:, 1]

    diff = torch.abs(target_margin - suspect_margin)

    return 1.0 / (1.0 + diff)


def batch_confidence_similarity(target_logits, suspect_logits):
    target_conf = F.softmax(target_logits, dim=1).max(dim=1).values
    suspect_conf = F.softmax(suspect_logits, dim=1).max(dim=1).values

    return 1.0 - torch.abs(target_conf - suspect_conf)


def safe_mean(values):
    if len(values) == 0:
        return 0.0
    return float(np.mean(values))


# ============================================================
# 6. Compute target-vs-suspect behavior similarity
# ============================================================

@torch.no_grad()
def compute_functional_features(
    target_model: nn.Module,
    suspect_model: nn.Module,
    loader: DataLoader,
    device: str,
    temperature: float,
) -> Dict[str, float]:

    logit_cosines = []
    prob_cosines = []
    inverse_kls = []
    symmetric_inverse_kls = []
    top1_agreements = []
    top5_overlaps = []
    margin_sims = []
    confidence_sims = []

    same_correct_values = []
    same_wrong_values = []

    target_conf_all = []
    suspect_conf_all = []

    target_pred_all = []
    suspect_pred_all = []
    labels_all = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        target_logits = target_model(images)
        suspect_logits = suspect_model(images)

        target_probs = F.softmax(target_logits, dim=1)
        suspect_probs = F.softmax(suspect_logits, dim=1)

        target_pred = target_logits.argmax(dim=1)
        suspect_pred = suspect_logits.argmax(dim=1)

        target_correct = target_pred == labels
        suspect_correct = suspect_pred == labels

        same_correct = target_correct & suspect_correct
        same_wrong = (~target_correct) & (~suspect_correct) & (target_pred == suspect_pred)

        logit_cosines.extend(
            batch_logit_cosine(target_logits, suspect_logits).detach().cpu().tolist()
        )
        prob_cosines.extend(
            batch_prob_cosine(target_logits, suspect_logits).detach().cpu().tolist()
        )
        inverse_kls.extend(
            batch_inverse_kl(target_logits, suspect_logits, temperature).detach().cpu().tolist()
        )
        symmetric_inverse_kls.extend(
            batch_symmetric_inverse_kl(target_logits, suspect_logits, temperature).detach().cpu().tolist()
        )
        top1_agreements.extend(
            batch_top1_agreement(target_logits, suspect_logits).detach().cpu().tolist()
        )
        top5_overlaps.extend(
            batch_top5_overlap(target_logits, suspect_logits).detach().cpu().tolist()
        )
        margin_sims.extend(
            batch_margin_similarity(target_logits, suspect_logits).detach().cpu().tolist()
        )
        confidence_sims.extend(
            batch_confidence_similarity(target_logits, suspect_logits).detach().cpu().tolist()
        )

        same_correct_values.extend(same_correct.float().detach().cpu().tolist())
        same_wrong_values.extend(same_wrong.float().detach().cpu().tolist())

        target_conf_all.extend(target_probs.max(dim=1).values.detach().cpu().tolist())
        suspect_conf_all.extend(suspect_probs.max(dim=1).values.detach().cpu().tolist())

        target_pred_all.extend(target_pred.detach().cpu().tolist())
        suspect_pred_all.extend(suspect_pred.detach().cpu().tolist())
        labels_all.extend(labels.detach().cpu().tolist())

    target_pred_all = np.array(target_pred_all)
    suspect_pred_all = np.array(suspect_pred_all)
    labels_all = np.array(labels_all)

    target_acc = float((target_pred_all == labels_all).mean())
    suspect_acc = float((suspect_pred_all == labels_all).mean())

    acc_similarity = 1.0 / (1.0 + abs(target_acc - suspect_acc))

    if len(target_conf_all) > 2:
        confidence_corr = float(np.corrcoef(target_conf_all, suspect_conf_all)[0, 1])
        if np.isnan(confidence_corr):
            confidence_corr = 0.0
    else:
        confidence_corr = 0.0

    features = {
        "logit_cosine": safe_mean(logit_cosines),
        "prob_cosine": safe_mean(prob_cosines),
        "inverse_kl": safe_mean(inverse_kls),
        "symmetric_inverse_kl": safe_mean(symmetric_inverse_kls),
        "top1_agreement": safe_mean(top1_agreements),
        "top5_overlap": safe_mean(top5_overlaps),
        "margin_similarity": safe_mean(margin_sims),
        "confidence_similarity": safe_mean(confidence_sims),
        "same_correct": safe_mean(same_correct_values),
        "same_wrong": safe_mean(same_wrong_values),
        "confidence_corr": confidence_corr,
        "target_acc": target_acc,
        "suspect_acc": suspect_acc,
        "acc_similarity": acc_similarity,
    }

    return features


# ============================================================
# 7. Suspect checkpoint finder
# ============================================================

def find_suspect_checkpoint(suspect_dir: Path, model_id: int) -> Path:
    candidates = [
        suspect_dir / f"{model_id}.safetensors",
        suspect_dir / f"model_{model_id}.safetensors",
        suspect_dir / f"suspect_{model_id:03d}.safetensors",
        suspect_dir / str(model_id) / "model.safetensors",
    ]

    for path in candidates:
        if path.exists():
            return path

    folder = suspect_dir / str(model_id)
    if folder.exists():
        matches = list(folder.rglob("*.safetensors"))
        if matches:
            return matches[0]

    raise FileNotFoundError(f"No .safetensors file found for suspect id {model_id}")


# ============================================================
# 8. Score variants
# ============================================================

def add_rank_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    for col in columns:
        df[f"{col}_rank"] = df[col].rank(method="average", pct=True)
    return df


def compute_scores(df: pd.DataFrame, variant: str) -> pd.DataFrame:
    feature_cols = [
        "logit_cosine",
        "prob_cosine",
        "inverse_kl",
        "symmetric_inverse_kl",
        "top1_agreement",
        "top5_overlap",
        "margin_similarity",
        "confidence_similarity",
        "same_correct",
        "same_wrong",
        "confidence_corr",
        "acc_similarity",
    ]

    df = add_rank_columns(df, feature_cols)

    if variant == "f1_logits_predictions":
        df["raw_score"] = (
            0.35 * df["logit_cosine_rank"] +
            0.25 * df["top1_agreement_rank"] +
            0.15 * df["top5_overlap_rank"] +
            0.15 * df["prob_cosine_rank"] +
            0.10 * df["symmetric_inverse_kl_rank"]
        )

    elif variant == "f2_kl_distillation":
        df["raw_score"] = (
            0.35 * df["symmetric_inverse_kl_rank"] +
            0.25 * df["inverse_kl_rank"] +
            0.20 * df["prob_cosine_rank"] +
            0.10 * df["logit_cosine_rank"] +
            0.10 * df["confidence_similarity_rank"]
        )

    elif variant == "f3_prediction_only":
        df["raw_score"] = (
            0.45 * df["top1_agreement_rank"] +
            0.25 * df["top5_overlap_rank"] +
            0.15 * df["same_wrong_rank"] +
            0.10 * df["same_correct_rank"] +
            0.05 * df["acc_similarity_rank"]
        )

    elif variant == "f4_confidence_margin":
        df["raw_score"] = (
            0.30 * df["margin_similarity_rank"] +
            0.25 * df["confidence_similarity_rank"] +
            0.20 * df["confidence_corr_rank"] +
            0.15 * df["prob_cosine_rank"] +
            0.10 * df["symmetric_inverse_kl_rank"]
        )

    elif variant == "f5_functional_ensemble":
        logit_score = (
            0.40 * df["logit_cosine_rank"] +
            0.40 * df["prob_cosine_rank"] +
            0.20 * df["symmetric_inverse_kl_rank"]
        )

        pred_score = (
            0.50 * df["top1_agreement_rank"] +
            0.40 * df["top5_overlap_rank"] +
            0.10 * df["same_wrong_rank"]
        )

        conf_score = (
            0.40 * df["confidence_similarity_rank"] +
            0.40 * df["margin_similarity_rank"] +
            0.20 * df["confidence_corr_rank"]
        )

        df["raw_score"] = (
            0.50 * logit_score +
            0.35 * pred_score +
            0.15 * conf_score
        )

    elif variant == "f6_max_signal":
        evidence_cols = [
            "logit_cosine_rank",
            "prob_cosine_rank",
            "symmetric_inverse_kl_rank",
            "top1_agreement_rank",
            "top5_overlap_rank",
            "same_wrong_rank",
        ]

        df["raw_score"] = df[evidence_cols].max(axis=1)

    else:
        raise ValueError(f"Unknown variant: {variant}")

    # Rank-normalized because leaderboard metric is ranking-based
    df["score"] = df["raw_score"].rank(method="average", pct=True)

    return df


# ============================================================
# 9. Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--suspect_dir", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--train_idx", type=str, required=True)

    parser.add_argument(
        "--split",
        type=str,
        default="train_target",
        choices=[
            "test",
            "train_target",
            "train_non_target",
            "train_all",
        ],
    )

    parser.add_argument(
        "--transform",
        type=str,
        default="target_aug",
        choices=[
            "clean",
            "target_aug",
        ],
    )

    parser.add_argument(
        "--variant",
        type=str,
        default="f1_logits_predictions",
        choices=[
            "f1_logits_predictions",
            "f2_kl_distillation",
            "f3_prediction_only",
            "f4_confidence_margin",
            "f5_functional_ensemble",
            "f6_max_signal",
        ],
    )

    parser.add_argument("--max_samples", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_models", type=int, default=360)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default="outputs_functional")

    args = parser.parse_args()

    set_seed(args.seed)

    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_name = (
        f"{args.split}_{args.transform}"
        f"_n{args.max_samples}_temp{args.temperature}_seed{args.seed}"
    )

    features_path = out_dir / f"features_{run_name}.csv"
    scored_path = out_dir / f"scored_{run_name}.csv"
    submission_path = out_dir / f"submission_{run_name}.csv"

    print("=" * 80)
    print("Functional Similarity Experiment")
    print("=" * 80)
    print(f"Target:       {args.target}")
    print(f"Suspects:     {args.suspect_dir}")
    print(f"Data root:    {args.data_root}")
    print(f"Train idx:    {args.train_idx}")
    print(f"Split:        {args.split}")
    print(f"Transform:    {args.transform}")
    print(f"Variant:      {args.variant}")
    print(f"Max samples:  {args.max_samples}")
    print(f"Temperature:  {args.temperature}")
    print(f"Device:       {device}")
    print("=" * 80)

    print("\nLoading target model...")
    target_model = load_model(args.target, device)

    print("\nBuilding CIFAR-100 probe loader...")
    loader = build_loader(
        data_root=args.data_root,
        split=args.split,
        transform_type=args.transform,
        train_idx_path=args.train_idx,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_samples=args.max_samples,
        seed=args.seed,
    )

    rows = []

    for model_id in range(args.num_models):
        print(f"\nProcessing suspect model {model_id}")

        try:
            suspect_path = find_suspect_checkpoint(Path(args.suspect_dir), model_id)
            print(f"  checkpoint: {suspect_path}")

            suspect_model = load_model(str(suspect_path), device)

            features = compute_functional_features(
                target_model=target_model,
                suspect_model=suspect_model,
                loader=loader,
                device=device,
                temperature=args.temperature,
            )

            row = {
                "id": model_id,
                "checkpoint_path": str(suspect_path),
                **features,
                "load_error": "",
            }

            del suspect_model

            if device == "cuda":
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"  ERROR: {e}")

            row = {
                "id": model_id,
                "checkpoint_path": "",
                "logit_cosine": 0.0,
                "prob_cosine": 0.0,
                "inverse_kl": 0.0,
                "symmetric_inverse_kl": 0.0,
                "top1_agreement": 0.0,
                "top5_overlap": 0.0,
                "margin_similarity": 0.0,
                "confidence_similarity": 0.0,
                "same_correct": 0.0,
                "same_wrong": 0.0,
                "confidence_corr": 0.0,
                "target_acc": 0.0,
                "suspect_acc": 0.0,
                "acc_similarity": 0.0,
                "load_error": str(e),
            }

        rows.append(row)

    df = pd.DataFrame(rows)

    # Clean numeric values
    for col in df.columns:
        if col not in ["checkpoint_path", "load_error"]:
            df[col] = (
                pd.to_numeric(df[col], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
            )

    df.to_csv(features_path, index=False)

    scored_df = compute_scores(df.copy(), args.variant)
    scored_df.to_csv(scored_path, index=False)

    submission_df = scored_df[["id", "score"]].copy()
    submission_df = submission_df.sort_values("id")
    submission_df.to_csv(submission_path, index=False)

    print("\nSaved files:")
    print(f"  Features:   {features_path}")
    print(f"  Scored:     {scored_path}")
    print(f"  Submission: {submission_path}")

    print("\nTop 30 suspects:")
    display_cols = [
        "id",
        "score",
        "raw_score",
        "logit_cosine",
        "prob_cosine",
        "symmetric_inverse_kl",
        "top1_agreement",
        "top5_overlap",
        "same_wrong",
        "confidence_corr",
        "target_acc",
        "suspect_acc",
    ]

    print(
        scored_df.sort_values("score", ascending=False)[display_cols]
        .head(30)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()