import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import resnet18
from safetensors.torch import load_file


# -----------------------------
# 1. Model architecture
# -----------------------------

def make_model():
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 100)
    return model


def load_model(checkpoint_path: str, device: str = "cpu"):
    model = make_model()
    state_dict = load_file(checkpoint_path, device=device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model.to(device)


# -----------------------------
# 2. Dataset helpers
# -----------------------------

def get_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5071, 0.4867, 0.4408),
            (0.2675, 0.2565, 0.2761)
        ),
    ])


def get_probe_loaders(data_root: str, train_idx_path: str, batch_size: int = 256):
    """
    Returns two DataLoaders:
      - train_loader: the exact training samples used by the target model
      - test_loader:  the full CIFAR-100 test split (unseen during target training)
    """
    transform = get_transform()

    # Full CIFAR-100 train split — we will sub-select by known indices
    train_dataset = datasets.CIFAR100(
        root=data_root, train=True, download=True, transform=transform
    )

    # Test split — target model never trained on this
    test_dataset = datasets.CIFAR100(
        root=data_root, train=False, download=True, transform=transform
    )

    with open(train_idx_path, "r") as f:
        train_indices = json.load(f)

    # Subset of training data the target model actually saw
    train_subset = Subset(train_dataset, train_indices)

    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    return train_loader, test_loader


# -----------------------------
# 3. Collect logits / probs
# -----------------------------

@torch.no_grad()
def collect_probs(model, loader, device):
    """
    Run the model over all batches in the loader.
    Returns a tensor of shape (N, 100) with softmax probabilities.
    """
    all_probs = []
    for images, _ in loader:
        images = images.to(device)
        logits = model(images)
        probs = F.softmax(logits, dim=-1)
        all_probs.append(probs.cpu())
    return torch.cat(all_probs, dim=0)  # (N, 100)


# -----------------------------
# 4. Per-comparison signals
# -----------------------------

def top1_agreement(target_probs: torch.Tensor, suspect_probs: torch.Tensor) -> float:
    """
    Fraction of samples where both models predict the same top-1 class.
    High agreement on test set → suspicious.
    """
    target_preds = target_probs.argmax(dim=-1)
    suspect_preds = suspect_probs.argmax(dim=-1)
    return (target_preds == suspect_preds).float().mean().item()


def mean_kl_divergence(target_probs: torch.Tensor, suspect_probs: torch.Tensor) -> float:
    """
    Average KL(target || suspect) over all samples.
    Lower KL → distributions are more similar → more suspicious.
    We return the negative so that higher = more stolen.
    """
    # Clamp to avoid log(0)
    target_probs = target_probs.clamp(min=1e-8)
    suspect_probs = suspect_probs.clamp(min=1e-8)
    kl = (target_probs * (target_probs.log() - suspect_probs.log())).sum(dim=-1)
    return kl.mean().item()


def soft_cosine_similarity(target_probs: torch.Tensor, suspect_probs: torch.Tensor) -> float:
    """
    Average cosine similarity between soft probability vectors.
    Stolen/distilled models will have very similar probability profiles.
    """
    # Normalise each row
    target_norm = F.normalize(target_probs, p=2, dim=-1)
    suspect_norm = F.normalize(suspect_probs, p=2, dim=-1)
    cos_per_sample = (target_norm * suspect_norm).sum(dim=-1)  # (N,)
    return cos_per_sample.mean().item()


def rank_correlation(target_probs: torch.Tensor, suspect_probs: torch.Tensor) -> float:
    """
    Average Spearman-like rank correlation between full probability vectors.
    Stolen models tend to preserve the relative ordering of class probabilities,
    not just the top-1.
    Uses Pearson on ranks as a fast approximation.
    """
    def to_ranks(t):
        # argsort twice gives rank
        return t.argsort(dim=-1).argsort(dim=-1).float()

    target_ranks = to_ranks(target_probs)
    suspect_ranks = to_ranks(suspect_probs)

    # Center
    target_c = target_ranks - target_ranks.mean(dim=-1, keepdim=True)
    suspect_c = suspect_ranks - suspect_ranks.mean(dim=-1, keepdim=True)

    num = (target_c * suspect_c).sum(dim=-1)
    denom = (
        target_c.norm(dim=-1) * suspect_c.norm(dim=-1)
    ).clamp(min=1e-8)

    rho_per_sample = num / denom
    return rho_per_sample.mean().item()


def hard_sample_agreement(
    target_probs: torch.Tensor,
    suspect_probs: torch.Tensor,
    confidence_threshold: float = 0.5,
) -> float:
    """
    Top-1 agreement restricted to samples where the TARGET model is uncertain
    (max probability < threshold).  Independent models diverge here; stolen
    models tend to mirror the target even on ambiguous inputs.
    """
    target_max_conf = target_probs.max(dim=-1).values
    hard_mask = target_max_conf < confidence_threshold

    if hard_mask.sum() == 0:
        return 0.0

    target_preds = target_probs[hard_mask].argmax(dim=-1)
    suspect_preds = suspect_probs[hard_mask].argmax(dim=-1)
    return (target_preds == suspect_preds).float().mean().item()


def memorisation_gap(
    suspect_train_probs: torch.Tensor,
    suspect_test_probs: torch.Tensor,
    target_train_labels: torch.Tensor,
    target_test_labels: torch.Tensor,
) -> float:
    """
    Measures how much more confident the suspect model is on the TARGET'S
    training indices compared to the test set.

    A model stolen/fine-tuned from the target inherits its memorisation:
    it will be more confident on those exact training points than an
    independent model would be.

    Returns: train_confidence - test_confidence  (higher → more suspicious)
    """
    train_conf = suspect_train_probs.max(dim=-1).values.mean().item()
    test_conf = suspect_test_probs.max(dim=-1).values.mean().item()
    return train_conf - test_conf


# -----------------------------
# 5. Aggregate into one score
# -----------------------------

def behavior_score(signals: dict) -> float:
    """
    Combine behavioral signals into a single stealing confidence score.

    Convention: all input signals are oriented so that HIGHER = more stolen.
      - top1_agreement_test:   high → stolen
      - soft_cosine_test:      high → stolen
      - rank_corr_test:        high → stolen
      - hard_sample_agree_test:high → stolen
      - neg_kl_test:           high (less negative) → stolen
      - memorisation_gap:      high → stolen
    """
    score = (
        0.25 * signals["top1_agreement_test"]
        + 0.20 * signals["soft_cosine_test"]
        + 0.20 * signals["rank_corr_test"]
        + 0.15 * signals["hard_sample_agree_test"]
        + 0.10 * signals["neg_kl_test"]       # already negated below
        + 0.10 * signals["memorisation_gap"]
    )
    return score


# -----------------------------
# 6. Main pipeline
# -----------------------------

def main():
    TARGET_CHECKPOINT = "./target_model/weights.safetensors"
    SUSPECT_DIR = "./suspect_models/"
    DATA_ROOT = "./data/"
    TRAIN_IDX_PATH = "./target_model/train_main_idx.json"

    OUTPUT_FEATURES = "behavior_features.csv"
    OUTPUT_SUBMISSION = "submission_behavior.csv"

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    # --- Load target model ---
    print("Loading target model...")
    target_model = load_model(TARGET_CHECKPOINT, device=DEVICE)

    # --- Build probe datasets ---
    print("Building probe loaders...")
    train_loader, test_loader = get_probe_loaders(DATA_ROOT, TRAIN_IDX_PATH)

    # --- Collect target probabilities once (reused for every suspect) ---
    print("Collecting target model probabilities...")
    target_train_probs = collect_probs(target_model, train_loader, DEVICE)
    target_test_probs = collect_probs(target_model, test_loader, DEVICE)

    # Ground-truth labels (for memorisation gap; probs already collected)
    # We need labels separately to pass to memorisation_gap helper
    train_labels = torch.tensor([
        train_loader.dataset.dataset.targets[i]
        for i in train_loader.dataset.indices
    ])
    test_labels = torch.tensor(test_loader.dataset.targets)

    rows = []
    raw_scores = []

    for model_id in range(360):
        suspect_path = Path(SUSPECT_DIR) / f"suspect_{model_id:03d}.safetensors"

        if not suspect_path.exists():
            raise FileNotFoundError(f"Suspect model not found: {suspect_path}")

        print(f"Processing suspect model {model_id}...")
        suspect_model = load_model(str(suspect_path), device=DEVICE)

        # Collect suspect probs on both splits
        suspect_train_probs = collect_probs(suspect_model, train_loader, DEVICE)
        suspect_test_probs = collect_probs(suspect_model, test_loader, DEVICE)

        # --- Compute all signals on the TEST split ---
        # (test split is unseen by the target → stronger generalisation signal)
        t1_test = top1_agreement(target_test_probs, suspect_test_probs)
        kl_test = mean_kl_divergence(target_test_probs, suspect_test_probs)
        cos_test = soft_cosine_similarity(target_test_probs, suspect_test_probs)
        rk_test = rank_correlation(target_test_probs, suspect_test_probs)
        hard_test = hard_sample_agreement(target_test_probs, suspect_test_probs)

        # Memorisation gap (uses train split confidence vs test split confidence)
        mem_gap = memorisation_gap(
            suspect_train_probs, suspect_test_probs,
            train_labels, test_labels
        )

        signals = {
            "top1_agreement_test": t1_test,
            "neg_kl_test": -kl_test,   # negate so higher = more similar
            "soft_cosine_test": cos_test,
            "rank_corr_test": rk_test,
            "hard_sample_agree_test": hard_test,
            "memorisation_gap": mem_gap,
        }

        score = behavior_score(signals)

        row = {"id": model_id, **signals, "raw_score": score}
        rows.append(row)
        raw_scores.append(score)

        # Free GPU memory between suspects
        del suspect_model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    # --- Normalise scores to [0, 1] ---
    scores_tensor = torch.tensor(raw_scores, dtype=torch.float32)
    min_s, max_s = scores_tensor.min().item(), scores_tensor.max().item()
    if max_s - min_s > 1e-8:
        norm_scores = ((scores_tensor - min_s) / (max_s - min_s)).tolist()
    else:
        norm_scores = [0.5] * len(raw_scores)

    for row, ns in zip(rows, norm_scores):
        row["score"] = ns

    # --- Save outputs ---
    features_df = pd.DataFrame(rows)
    features_df.to_csv(OUTPUT_FEATURES, index=False)

    submission_df = features_df[["id", "score"]]
    submission_df.to_csv(OUTPUT_SUBMISSION, index=False)

    print(f"Saved features to:   {OUTPUT_FEATURES}")
    print(f"Saved submission to: {OUTPUT_SUBMISSION}")


if __name__ == "__main__":
    main()
