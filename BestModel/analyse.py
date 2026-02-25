import os
import ast
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    precision_recall_curve, f1_score, average_precision_score
)

# ── Config ───────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(__file__)
DATA_DIR        = os.path.join(BASE_DIR, "..", "data")
PRED_DIR        = os.path.join(BASE_DIR, "..", "predictions")
OUTPUT_DIR      = os.path.join(BASE_DIR, "..", "eda")
BASELINE_F1     = 0.48          # RoBERTa baseline from the paper
THRESHOLD       = 0.12          # nominal threshold used for plots

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Load Data ─────────────────────────────────────────────────────────────────
def load_dev_with_predictions():
    """
    Load dev set + our latest predictions, using the *official* SemEval
    binarisation (>=2 annotators in the CSV vote labels).
    """
    # Load full TSV (for text / keyword / etc.)
    df = pd.read_csv(
        os.path.join(DATA_DIR, "dontpatronizeme_pcl.tsv"),
        sep="\t",
        header=None,
        names=["par_id", "art_id", "keyword", "country", "text", "label"],
        skiprows=4
    )
    df["par_id"]      = df["par_id"].astype(str).str.strip()
    df["token_count"] = df["text"].str.split().str.len()

    # Load dev split IDs + vote vectors and reproduce evaluation.py / train.py logic
    dev_ids = pd.read_csv(os.path.join(DATA_DIR, "dev_semeval_parids-labels.csv"))
    dev_ids["par_id"] = dev_ids["par_id"].astype(str).str.strip()
    dev_ids["binary_label"] = dev_ids["label"].apply(
        lambda x: 1 if sum(ast.literal_eval(x)) >= 2 else 0
    )

    # Filter to dev set and then merge in CSV order so we stay aligned with predictions
    dev_df = df[df["par_id"].isin(dev_ids["par_id"])].copy()
    dev_df = dev_ids[["par_id", "binary_label"]].merge(
        dev_df.drop(columns=["label"]), on="par_id", how="left"
    )

    # Load our predictions
    with open(os.path.join(PRED_DIR, "dev.txt")) as f:
        preds = [int(line.strip()) for line in f]

    assert len(preds) == len(dev_df), (
        f"Prediction count mismatch: {len(preds)} preds vs {len(dev_df)} dev rows"
    )

    dev_df["pred"] = preds
    dev_df["correct"] = (dev_df["pred"] == dev_df["binary_label"]).astype(int)

    return dev_df


# ── Section 1: Confusion Matrix ───────────────────────────────────────────────
def plot_confusion_matrix(dev_df, overall_f1):
    cm   = confusion_matrix(dev_df["binary_label"], dev_df["pred"])
    disp = ConfusionMatrixDisplay(cm, display_labels=["No PCL", "PCL"])

    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix — DeBERTa-v3-large (dev set)", fontsize=11)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")

    tn, fp, fn, tp = cm.ravel()
    print(f"\nTP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"Precision : {tp / (tp + fp):.4f}")
    print(f"Recall    : {tp / (tp + fn):.4f}")
    print(f"F1        : {overall_f1:.4f}  (baseline {BASELINE_F1})")
    return tp, fp, fn, tn


# ── Section 2: Precision-Recall Curve ────────────────────────────────────────
# We need raw probabilities for this — re-read from a saved probs file if it
# exists, otherwise approximate from binary preds (curve will be a single point)
def plot_pr_curve(dev_df):
    probs_path = os.path.join(PRED_DIR, "dev_probs.npy")

    if os.path.exists(probs_path):
        probs = np.load(probs_path)
        precision, recall, thresholds = precision_recall_curve(
            dev_df["binary_label"], probs
        )
        ap = average_precision_score(dev_df["binary_label"], probs)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(recall, precision, lw=2, label=f"DeBERTa-v3-large (AP={ap:.3f})")

        # Mark operating threshold
        idx = np.argmin(np.abs(thresholds - THRESHOLD))
        ax.scatter(recall[idx], precision[idx], s=100, zorder=5,
                   label=f"threshold={THRESHOLD}")

        # Baseline as a point (P=0.73, R=0.69 from paper Table 4 RoBERTa)
        ax.scatter(0.537, 0.665, marker="x", s=120, color="red",
                   label=f"RoBERTa baseline (F1={BASELINE_F1})")

        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curve (dev set, PCL class)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        out = os.path.join(OUTPUT_DIR, "pr_curve.png")
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"Saved: {out}")
    else:
        print("dev_probs.npy not found — skipping PR curve.")
        print("Re-run train.py with np.save('predictions/dev_probs.npy', dev_probs) added.")


# ── Section 3: Performance by Keyword Category ────────────────────────────────
def analyse_by_keyword(dev_df, overall_f1):
    results = []
    for kw, group in dev_df.groupby("keyword"):
        total   = len(group)
        pcl     = group["binary_label"].sum()
        tp      = ((group["pred"] == 1) & (group["binary_label"] == 1)).sum()
        fp      = ((group["pred"] == 1) & (group["binary_label"] == 0)).sum()
        fn      = ((group["pred"] == 0) & (group["binary_label"] == 1)).sum()
        prec    = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec     = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1      = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        results.append({
            "keyword":   kw,
            "total":     total,
            "pcl_count": pcl,
            "pcl_%":     round(100 * pcl / total, 1),
            "precision": round(prec, 3),
            "recall":    round(rec, 3),
            "f1":        round(f1, 3),
        })

    kw_df = pd.DataFrame(results).sort_values("f1")
    print("\n── Keyword-level Performance ──────────────────────────────────")
    print(kw_df.to_string(index=False))

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#d62728" if f < overall_f1 else "#2ca02c" for f in kw_df["f1"]]
    ax.barh(kw_df["keyword"], kw_df["f1"], color=colors)
    ax.axvline(overall_f1,   color="black",  lw=1.5, linestyle="--",
               label=f"Overall F1={overall_f1:.3f}")
    ax.axvline(BASELINE_F1,  color="red",    lw=1.5, linestyle=":",
               label=f"Baseline F1={BASELINE_F1}")
    ax.set_xlabel("F1 Score (PCL class)")
    ax.set_title("Per-Keyword F1 — DeBERTa-v3-large vs Baseline")
    ax.legend(fontsize=8)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "keyword_f1.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")

    return kw_df


# ── Section 4: Token Length vs Error Rate ─────────────────────────────────────
def analyse_length_vs_errors(dev_df):
    bins   = [0, 20, 40, 60, 80, 100, 200, 1000]
    labels = ["0-20", "21-40", "41-60", "61-80", "81-100", "101-200", "200+"]
    dev_df["length_bin"] = pd.cut(dev_df["token_count"], bins=bins, labels=labels)

    results = []
    for lb, group in dev_df.groupby("length_bin", observed=True):
        pcl_group = group[group["binary_label"] == 1]
        fn_rate   = (pcl_group["pred"] == 0).mean() if len(pcl_group) > 0 else 0.0
        fp_rate   = (group[group["binary_label"] == 0]["pred"] == 1).mean()
        results.append({
            "length_bin": lb,
            "n":          len(group),
            "pcl_n":      len(pcl_group),
            "fn_rate":    round(fn_rate, 3),
            "fp_rate":    round(fp_rate, 3),
        })

    len_df = pd.DataFrame(results)
    print("\n── Length-bin Error Analysis ───────────────────────────────────")
    print(len_df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(len_df))
    w = 0.35
    ax.bar(x - w/2, len_df["fn_rate"], w, label="False Negative Rate", color="#d62728")
    ax.bar(x + w/2, len_df["fp_rate"], w, label="False Positive Rate",  color="#1f77b4")
    ax.set_xticks(x)
    ax.set_xticklabels(len_df["length_bin"])
    ax.set_xlabel("Token Length Bin")
    ax.set_ylabel("Error Rate")
    ax.set_title("Error Rate by Text Length (dev set)")
    ax.legend()
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "length_error_rate.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


# ── Section 5: Error Analysis — False Positives & Negatives ──────────────────
def error_analysis(dev_df, n_examples=10):
    fp_df = dev_df[(dev_df["pred"] == 1) & (dev_df["binary_label"] == 0)].copy()
    fn_df = dev_df[(dev_df["pred"] == 0) & (dev_df["binary_label"] == 1)].copy()

    print(f"\n── Error Counts ────────────────────────────────────────────────")
    print(f"False Positives (predicted PCL, actually fine): {len(fp_df)}")
    print(f"False Negatives (missed PCL):                   {len(fn_df)}")

    print(f"\n── False Positives (sample of {n_examples}) ────────────────────")
    for _, row in fp_df.sample(min(n_examples, len(fp_df)), random_state=42).iterrows():
        print(f"  keyword={row['keyword']}")
        print(f"  text   ={row['text'][:200]}")
        print()

    print(f"\n── False Negatives (sample of {n_examples}) ────────────────────")
    for _, row in fn_df.sample(min(n_examples, len(fn_df)), random_state=42).iterrows():
        print(f"  keyword={row['keyword']}")
        print(f"  text   ={row['text'][:200]}")
        print()

    # Keyword breakdown of errors
    print("── False Positive breakdown by keyword ─────────────────────────")
    print(fp_df["keyword"].value_counts().to_string())
    print("\n── False Negative breakdown by keyword ─────────────────────────")
    print(fn_df["keyword"].value_counts().to_string())

    return fp_df, fn_df


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Loading dev set with predictions...")
    dev_df = load_dev_with_predictions()

    # Compute current F1 from latest predictions vs official dev labels
    overall_f1 = f1_score(dev_df["binary_label"], dev_df["pred"])

    print(f"\nDev set: {len(dev_df)} samples | PCL: {dev_df['binary_label'].sum()}")
    print(f"Our F1: {overall_f1:.4f}  |  Baseline F1: {BASELINE_F1}\n")

    print("=" * 60)
    print("1. Confusion Matrix")
    print("=" * 60)
    plot_confusion_matrix(dev_df, overall_f1)

    print("\n" + "=" * 60)
    print("2. Precision-Recall Curve")
    print("=" * 60)
    plot_pr_curve(dev_df)

    print("\n" + "=" * 60)
    print("3. Performance by Keyword")
    print("=" * 60)
    kw_df = analyse_by_keyword(dev_df, overall_f1)

    print("\n" + "=" * 60)
    print("4. Token Length vs Error Rate")
    print("=" * 60)
    analyse_length_vs_errors(dev_df)

    print("\n" + "=" * 60)
    print("5. Error Analysis")
    print("=" * 60)
    fp_df, fn_df = error_analysis(dev_df, n_examples=8)

    print("\nDone. All figures saved to", OUTPUT_DIR)


if __name__ == "__main__":
    main()