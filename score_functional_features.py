import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


def add_rank_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    for col in columns:
        if col not in df.columns:
            raise ValueError(f"Missing feature column: {col}")

        df[f"{col}_rank"] = (
            pd.to_numeric(df[col], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .rank(method="average", pct=True)
        )

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
            0.45 * df["logit_cosine_rank"] +
            0.30 * df["prob_cosine_rank"] +
            0.25 * df["symmetric_inverse_kl_rank"]
        )

        pred_score = (
            0.50 * df["top1_agreement_rank"] +
            0.30 * df["top5_overlap_rank"] +
            0.20 * df["same_wrong_rank"]
        )

        conf_score = (
            0.40 * df["confidence_similarity_rank"] +
            0.35 * df["margin_similarity_rank"] +
            0.25 * df["confidence_corr_rank"]
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

    elif variant == "f7_logit_only":
        df["raw_score"] = df["logit_cosine_rank"]

    elif variant == "f8_top1_only":
        df["raw_score"] = df["top1_agreement_rank"]

    elif variant == "f9_same_wrong_heavy":
        df["raw_score"] = (
            0.45 * df["same_wrong_rank"] +
            0.25 * df["top1_agreement_rank"] +
            0.15 * df["logit_cosine_rank"] +
            0.15 * df["symmetric_inverse_kl_rank"]
        )
    elif variant == "single_logit_cosine":
        df["raw_score"] = df["logit_cosine_rank"]

    elif variant == "single_prob_cosine":
        df["raw_score"] = df["prob_cosine_rank"]

    elif variant == "single_inverse_kl":
        df["raw_score"] = df["inverse_kl_rank"]

    elif variant == "single_sym_kl":
        df["raw_score"] = df["symmetric_inverse_kl_rank"]

    elif variant == "single_top1":
        df["raw_score"] = df["top1_agreement_rank"]

    elif variant == "single_top5":
        df["raw_score"] = df["top5_overlap_rank"]

    elif variant == "single_same_wrong":
        df["raw_score"] = df["same_wrong_rank"]

    elif variant == "single_confidence":
        df["raw_score"] = df["confidence_similarity_rank"]

    elif variant == "single_margin":
        df["raw_score"] = df["margin_similarity_rank"]

    elif variant == "single_conf_corr":
        df["raw_score"] = df["confidence_corr_rank"]
    
    elif variant == "simple_top5_top1":
        df["raw_score"] = (
            0.70 * df["top5_overlap_rank"] +
            0.30 * df["top1_agreement_rank"]
        )

    elif variant == "simple_top5_samewrong":
        df["raw_score"] = (
            0.75 * df["top5_overlap_rank"] +
            0.25 * df["same_wrong_rank"]
        )

    elif variant == "top5_samewrong_90_10":
        df["raw_score"] = (
            0.90 * df["top5_overlap_rank"] +
            0.10 * df["same_wrong_rank"]
        )

    elif variant == "top5_samewrong_80_20":
        df["raw_score"] = (
            0.80 * df["top5_overlap_rank"] +
            0.20 * df["same_wrong_rank"]
        )

    elif variant == "top5_samewrong_70_30":
        df["raw_score"] = (
            0.70 * df["top5_overlap_rank"] +
            0.30 * df["same_wrong_rank"]
        )

    elif variant == "top5_samewrong_60_40":
        df["raw_score"] = (
            0.60 * df["top5_overlap_rank"] +
            0.40 * df["same_wrong_rank"]
        )

    elif variant == "top5_samewrong_50_50":
        df["raw_score"] = (
            0.50 * df["top5_overlap_rank"] +
            0.50 * df["same_wrong_rank"]
        )

    elif variant == "top5_samewrong_top1":
        df["raw_score"] = (
            0.70 * df["top5_overlap_rank"] +
            0.20 * df["same_wrong_rank"] +
            0.10 * df["top1_agreement_rank"]
        )

    elif variant == "top5_top1_samewrong":
        df["raw_score"] = (
            0.75 * df["top5_overlap_rank"] +
            0.15 * df["top1_agreement_rank"] +
            0.10 * df["same_wrong_rank"]
        )

    elif variant == "max_top5_samewrong":
        df["raw_score"] = df[
            [
                "top5_overlap_rank",
                "same_wrong_rank",
            ]
        ].max(axis=1)

    elif variant == "max_top5_top1_samewrong":
        df["raw_score"] = df[
            [
                "top5_overlap_rank",
                "top1_agreement_rank",
                "same_wrong_rank",
            ]
        ].max(axis=1)

    else:
        raise ValueError(f"Unknown variant: {variant}")

    df["score"] = df["raw_score"].rank(method="average", pct=True)

    return df


def validate_submission(submission_df: pd.DataFrame):
    if list(submission_df.columns) != ["id", "score"]:
        raise ValueError("Submission must have exactly columns: id, score")

    if len(submission_df) != 360:
        raise ValueError(f"Submission must have 360 rows, got {len(submission_df)}")

    if submission_df["id"].min() != 0 or submission_df["id"].max() != 359:
        raise ValueError("IDs must be from 0 to 359")

    if submission_df["id"].nunique() != 360:
        raise ValueError("IDs must be unique")

    scores = pd.to_numeric(submission_df["score"], errors="coerce")

    if scores.isna().any():
        raise ValueError("Scores contain NaN or non-numeric values")

    if not np.isfinite(scores).all():
        raise ValueError("Scores contain Inf or -Inf")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--features",
        type=str,
        required=True,
        help="Path to saved functional features CSV",
    )

    parser.add_argument(
        "--variant",
        type=str,
        required=True,
        choices=[
            "f1_logits_predictions",
            "f2_kl_distillation",
            "f3_prediction_only",
            "f4_confidence_margin",
            "f5_functional_ensemble",
            "f6_max_signal",
            "f7_logit_only",
            "f8_top1_only",
            "f9_same_wrong_heavy",
            "single_logit_cosine",
            "single_prob_cosine",
            "single_inverse_kl",
            "single_sym_kl",
            "single_top1",
            "single_top5",
            "single_same_wrong",
            "single_confidence",
            "single_margin",
            "single_conf_corr",
            "simple_top5_top1",
            "simple_top5_samewrong",
            "top5_samewrong_90_10",
            "top5_samewrong_80_20",
            "top5_samewrong_70_30",
            "top5_samewrong_top1",
            "max_top5_samewrong",
        ],
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default="outputs_functional_scored",
    )

    args = parser.parse_args()

    features_path = Path(args.features)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(features_path)

    scored_df = compute_scores(df.copy(), args.variant)

    stem = features_path.stem
    scored_path = out_dir / f"scored_{stem}_{args.variant}.csv"
    submission_path = out_dir / f"submission_{stem}_{args.variant}.csv"

    scored_df.to_csv(scored_path, index=False)

    submission_df = scored_df[["id", "score"]].copy()
    submission_df = submission_df.sort_values("id").reset_index(drop=True)

    validate_submission(submission_df)

    submission_df.to_csv(submission_path, index=False)

    print(f"Saved scored features: {scored_path}")
    print(f"Saved submission:      {submission_path}")

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
    ]
    display_cols = [c for c in display_cols if c in scored_df.columns]

    print(
        scored_df.sort_values("score", ascending=False)[display_cols]
        .head(30)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()