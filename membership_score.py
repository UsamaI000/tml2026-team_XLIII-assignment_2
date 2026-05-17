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


def get_loaders(data_root: str, train_idx_path: str, batch_size: int = 256):
    """
    Returns three DataLoaders:
      - member_loader:     exact training samples the target was trained on
      - nonmember_loader:  CIFAR-100 train samples the target was NOT trained on
      - test_loader:       full CIFAR-100 test split (always non-member)

    Having two non-member sets lets us cross-check signals.
    """
    transform = get_transform()

    full_train = datasets.CIFAR100(
        root=data_root, train=True, download=True, transform=transform
    )
    test_dataset = datasets.CIFAR100(
        root=data_root, train=False, download=True, transform=transform
    )

    with open(train_idx_path, "r") as f:
        member_indices = set(json.load(f))

    all_indices = set(range(len(full_train)))
    nonmember_indices = list(all_indices - member_indices)
    member_indices = list(member_indices)

    # Use the same count for non-members to keep comparisons balanced
    nonmember_indices = nonmember_indices[:len(member_indices)]

    member_loader = DataLoader(
        Subset(full_train, member_indices),
        batch_size=batch_size, shuffle=False, num_workers=2
    )
    nonmember_loader = DataLoader(
        Subset(full_train, nonmember_indices),
        batch_size=batch_size, shuffle=False, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size, shuffle=False, num_workers=2
    )

    return member_loader, nonmember_loader, test_loader


# -----------------------------
# 3. Per-sample statistics
# -----------------------------

@torch.no_grad()
def collect_stats(model, loader, device):
    """
    For every sample in the loader, collect:
      - loss:          cross-entropy loss (lower for memorised samples)
      - confidence:    max softmax probability
      - correct:       whether top-1 prediction matches true label
      - entropy:       entropy of the softmax distribution (lower = more certain)
      - margin:        gap between top-1 and top-2 probability (higher = more certain)

    Returns a dict of tensors, each of shape (N,).
    """
    all_loss = []
    all_conf = []
    all_correct = []
    all_entropy = []
    all_margin = []

    criterion = nn.CrossEntropyLoss(reduction="none")

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        probs = F.softmax(logits, dim=-1)

        loss = criterion(logits, labels)

        conf, preds = probs.max(dim=-1)
        correct = (preds == labels).float()

        entropy = -(probs * (probs + 1e-8).log()).sum(dim=-1)

        top2 = probs.topk(2, dim=-1).values
        margin = top2[:, 0] - top2[:, 1]

        all_loss.append(loss.cpu())
        all_conf.append(conf.cpu())
        all_correct.append(correct.cpu())
        all_entropy.append(entropy.cpu())
        all_margin.append(margin.cpu())

    return {
        "loss":      torch.cat(all_loss),
        "confidence": torch.cat(all_conf),
        "correct":   torch.cat(all_correct),
        "entropy":   torch.cat(all_entropy),
        "margin":    torch.cat(all_margin),
    }


# -----------------------------
# 4. Membership signals
# -----------------------------

def gap_signal(member_vals: torch.Tensor, nonmember_vals: torch.Tensor) -> float:
    """
    Returns mean(member) - mean(nonmember).
    For confidence/correct/margin: positive gap → model is more certain on members.
    For loss/entropy: we negate so the convention is always higher = more member-like.
    """
    return member_vals.mean().item() - nonmember_vals.mean().item()


def loss_gap(member_stats, nonmember_stats) -> float:
    """
    Non-members should have HIGHER loss than members for any well-trained model.
    But a stolen model inherits the target's memorisation, so its loss gap on the
    TARGET's training indices will closely mirror the target's own loss gap.

    We return:  mean_loss(nonmember) - mean_loss(member)
    Higher → suspect is more confident on exactly the target's training set.
    """
    return (
        nonmember_stats["loss"].mean().item()
        - member_stats["loss"].mean().item()
    )


def confidence_gap(member_stats, nonmember_stats) -> float:
    """Higher confidence on members than non-members → stolen."""
    return gap_signal(member_stats["confidence"], nonmember_stats["confidence"])


def accuracy_gap(member_stats, nonmember_stats) -> float:
    """Higher accuracy on members than non-members → stolen."""
    return gap_signal(member_stats["correct"], nonmember_stats["correct"])


def entropy_gap(member_stats, nonmember_stats) -> float:
    """
    Members should have LOWER entropy (model is more certain).
    Return nonmember_entropy - member_entropy so higher = more stolen.
    """
    return (
        nonmember_stats["entropy"].mean().item()
        - member_stats["entropy"].mean().item()
    )


def margin_gap(member_stats, nonmember_stats) -> float:
    """Higher margin (top1 - top2 prob) on members → stolen."""
    return gap_signal(member_stats["margin"], nonmember_stats["margin"])


def membership_inference_auc(
    member_stats: dict,
    nonmember_stats: dict,
    signal: str = "loss",
    higher_is_member: bool = False,
) -> float:
    """
    Computes the ROC-AUC of a simple threshold membership inference attack
    using a single signal (loss, confidence, entropy, margin).

    AUC > 0.5 means the signal separates members from non-members.
    A stolen model will have AUC close to the TARGET model's AUC on the
    same membership split, because it inherited the same memorisation.

    higher_is_member: if True, higher signal value → predicts membership.
    """
    from torchmetrics.functional import auroc  # soft dep, falls back below

    member_scores = member_stats[signal]
    nonmember_scores = nonmember_stats[signal]

    # 1 = member, 0 = non-member
    labels = torch.cat([
        torch.ones(len(member_scores)),
        torch.zeros(len(nonmember_scores))
    ]).long()

    scores = torch.cat([member_scores, nonmember_scores])
    if not higher_is_member:
        scores = -scores  # flip so that lower loss = higher member score

    try:
        auc = auroc(scores, labels, task="binary").item()
    except Exception:
        # Fallback: manual AUC via sorted ranks
        n_pos = labels.sum().item()
        n_neg = len(labels) - n_pos
        if n_pos == 0 or n_neg == 0:
            return 0.5
        sorted_idx = scores.argsort(descending=True)
        sorted_labels = labels[sorted_idx].float()
        tp_cumsum = sorted_labels.cumsum(0)
        fp_cumsum = (1 - sorted_labels).cumsum(0)
        tpr = tp_cumsum / n_pos
        fpr = fp_cumsum / n_neg
        # Trapezoidal rule
        auc = torch.trapz(tpr, fpr).abs().item()

    return auc


def gap_similarity_to_target(
    target_member_stats: dict,
    target_nonmember_stats: dict,
    suspect_member_stats: dict,
    suspect_nonmember_stats: dict,
    signal: str,
    higher_is_member: bool = True,
) -> float:
    """
    The KEY membership signal for stealing detection:

    The target model has a characteristic memorisation gap on its own
    training data.  A stolen model will exhibit a SIMILAR gap because it
    was initialised from (or distilled from) the target.  An independent
    model will have a gap shaped by its OWN training data, not the target's.

    We measure:  1 - |gap_target - gap_suspect| / (|gap_target| + eps)

    Returns a value in [0, 1].  Higher = suspect gap matches target gap = more stolen.
    """
    def compute_gap(m_stats, nm_stats):
        if higher_is_member:
            return m_stats[signal].mean().item() - nm_stats[signal].mean().item()
        else:
            return nm_stats[signal].mean().item() - m_stats[signal].mean().item()

    target_gap = compute_gap(target_member_stats, target_nonmember_stats)
    suspect_gap = compute_gap(suspect_member_stats, suspect_nonmember_stats)

    diff = abs(target_gap - suspect_gap)
    norm = abs(target_gap) + 1e-8
    similarity = max(0.0, 1.0 - diff / norm)
    return similarity


# -----------------------------
# 5. Aggregate into one score
# -----------------------------

def membership_score(signals: dict) -> float:
    """
    Combine membership signals into a single stealing confidence score.
    All signals oriented so that HIGHER = more stolen.
    """
    score = (
        0.25 * signals["loss_gap"]
        + 0.20 * signals["confidence_gap"]
        + 0.15 * signals["entropy_gap"]
        + 0.15 * signals["margin_gap"]
        + 0.10 * signals["accuracy_gap"]
        + 0.10 * signals["gap_sim_loss"]       # gap similarity to target (loss)
        + 0.05 * signals["gap_sim_confidence"]  # gap similarity to target (confidence)
    )
    return score


# -----------------------------
# 6. Main pipeline
# -----------------------------

def main():
    BASE = Path(__file__).parent
    TARGET_CHECKPOINT = BASE / "target_model/weights.safetensors"
    SUSPECT_DIR = BASE / "suspect_models/"
    DATA_ROOT = BASE / "data/"
    TRAIN_IDX_PATH = BASE / "target_model/train_main_idx.json"

    OUTPUT_FEATURES = BASE / "membership_features.csv"
    OUTPUT_SUBMISSION = BASE / "submission_membership.csv"

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    # --- Load target model ---
    print("Loading target model...")
    target_model = load_model(TARGET_CHECKPOINT, device=DEVICE)

    # --- Build loaders ---
    print("Building probe loaders...")
    member_loader, nonmember_loader, test_loader = get_loaders(DATA_ROOT, TRAIN_IDX_PATH)

    # --- Collect TARGET model stats once ---
    # These define the "reference memorisation fingerprint" we compare suspects against
    print("Collecting target model membership stats...")
    target_member_stats = collect_stats(target_model, member_loader, DEVICE)
    target_nonmember_stats = collect_stats(target_model, nonmember_loader, DEVICE)

    print(f"  Target loss gap (nonmember - member): "
          f"{loss_gap(target_member_stats, target_nonmember_stats):.4f}")
    print(f"  Target confidence gap (member - nonmember): "
          f"{confidence_gap(target_member_stats, target_nonmember_stats):.4f}")

    rows = []
    raw_scores = []

    for model_id in range(360):
        suspect_path = Path(SUSPECT_DIR) / f"suspect_{model_id:03d}.safetensors"

        if not suspect_path.exists():
            raise FileNotFoundError(f"Suspect model not found: {suspect_path}")

        print(f"Processing suspect model {model_id}...")
        suspect_model = load_model(str(suspect_path), device=DEVICE)

        suspect_member_stats = collect_stats(suspect_model, member_loader, DEVICE)
        suspect_nonmember_stats = collect_stats(suspect_model, nonmember_loader, DEVICE)

        # --- Raw gap signals ---
        # These measure how much more the suspect "knows" the target's training data
        sig_loss_gap        = loss_gap(suspect_member_stats, suspect_nonmember_stats)
        sig_confidence_gap  = confidence_gap(suspect_member_stats, suspect_nonmember_stats)
        sig_accuracy_gap    = accuracy_gap(suspect_member_stats, suspect_nonmember_stats)
        sig_entropy_gap     = entropy_gap(suspect_member_stats, suspect_nonmember_stats)
        sig_margin_gap      = margin_gap(suspect_member_stats, suspect_nonmember_stats)

        # --- Gap similarity to target ---
        # Does the suspect's memorisation fingerprint MATCH the target's?
        sig_gap_sim_loss = gap_similarity_to_target(
            target_member_stats, target_nonmember_stats,
            suspect_member_stats, suspect_nonmember_stats,
            signal="loss", higher_is_member=False   # lower loss = member
        )
        sig_gap_sim_conf = gap_similarity_to_target(
            target_member_stats, target_nonmember_stats,
            suspect_member_stats, suspect_nonmember_stats,
            signal="confidence", higher_is_member=True
        )

        # --- Membership inference AUC ---
        # How well does a threshold on loss separate members from non-members?
        # Stolen models tend to have similar AUC to the target on the same split.
        suspect_mi_auc = membership_inference_auc(
            suspect_member_stats, suspect_nonmember_stats,
            signal="loss", higher_is_member=False
        )
        target_mi_auc = membership_inference_auc(
            target_member_stats, target_nonmember_stats,
            signal="loss", higher_is_member=False
        )
        # Closeness of AUC to target AUC (another gap-similarity signal)
        sig_auc_similarity = 1.0 - abs(suspect_mi_auc - target_mi_auc)

        signals = {
            "loss_gap":           sig_loss_gap,
            "confidence_gap":     sig_confidence_gap,
            "accuracy_gap":       sig_accuracy_gap,
            "entropy_gap":        sig_entropy_gap,
            "margin_gap":         sig_margin_gap,
            "gap_sim_loss":       sig_gap_sim_loss,
            "gap_sim_confidence": sig_gap_sim_conf,
            "suspect_mi_auc":     suspect_mi_auc,
            "target_mi_auc":      target_mi_auc,
            "auc_similarity":     sig_auc_similarity,
        }

        score = membership_score(signals)

        row = {"id": model_id, **signals, "raw_score": score}
        rows.append(row)
        raw_scores.append(score)

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

    # --- Save ---
    features_df = pd.DataFrame(rows)
    features_df.to_csv(OUTPUT_FEATURES, index=False)

    submission_df = features_df[["id", "score"]]
    submission_df.to_csv(OUTPUT_SUBMISSION, index=False)

    print(f"Saved features to:   {OUTPUT_FEATURES}")
    print(f"Saved submission to: {OUTPUT_SUBMISSION}")


if __name__ == "__main__":
    main()
