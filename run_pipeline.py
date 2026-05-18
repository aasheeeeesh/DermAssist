"""
DermAssist Full Pipeline — Phase 1, 2 & 3
==========================================
Phase 1 : Generate REALISTIC dummy skin lesion data (with noise + class overlap)
Phase 2 : Extract CNN feature embeddings using ResNet50
Phase 3 : Train ALL 5 classifiers, compare, and save best model

Why was accuracy 100% before?
  The old dummy data used completely different RGB ranges per class (benign=warm,
  melanoma=dark). ResNet50 trivially separated them. Now both classes share the
  same base color distribution — only subtle texture/patch differences exist —
  forcing the classifiers to actually learn rather than memorise.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (no GUI window needed)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image

# ── Ensure the project root is on sys.path ────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.config import paths, preprocessing_config, model_config
from src.preprocessing import get_dataloaders
from src.embeddings import get_feature_extractor, extract_embeddings, save_embeddings, load_embeddings
from src.models import get_classifier, train_classifier, predict_classifier
from src.evaluation import evaluate_classifier, plot_confusion_matrix, plot_roc_curve

SEP = "\n" + "=" * 65

# ─────────────────────────────────────────────────────────────────────────────
#  PHASE 1: DATA SETUP — Realistic dummy dataset WITH class overlap + noise
# ─────────────────────────────────────────────────────────────────────────────
def phase1_data_setup(force=False):
    print(SEP)
    print("  PHASE 1 — DATA SETUP  (Real ISIC Dataset)")
    print(SEP)

    data_dir = paths.DATA_DIR
    csv_path = data_dir / "metadata.csv"
    raw_dir = data_dir / "raw" / "melanoma_cancer_dataset"

    if not raw_dir.exists():
        print(f"[ERROR] Real ISIC dataset raw folder not found at {raw_dir}!")
        sys.exit(1)

    print("Scanning downloaded raw dataset...")
    import glob

    train_benign = glob.glob(str(raw_dir / "train" / "benign" / "*.jpg"))
    train_malignant = glob.glob(str(raw_dir / "train" / "malignant" / "*.jpg"))
    test_benign = glob.glob(str(raw_dir / "test" / "benign" / "*.jpg"))
    test_malignant = glob.glob(str(raw_dir / "test" / "malignant" / "*.jpg"))

    print(f"Original Dataset counts:")
    print(f"  Train: Benign={len(train_benign)}, Malignant={len(train_malignant)}")
    print(f"  Test:  Benign={len(test_benign)}, Malignant={len(test_malignant)}")

    # Sample a representative subset (300 train of each class, 50 test of each class)
    # to run extremely fast on CPU (total ~700 images) while remaining 100% real.
    rng = np.random.default_rng(42)
    selected_train_b = rng.choice(train_benign, min(300, len(train_benign)), replace=False)
    selected_train_m = rng.choice(train_malignant, min(300, len(train_malignant)), replace=False)
    selected_test_b = rng.choice(test_benign, min(50, len(test_benign)), replace=False)
    selected_test_m = rng.choice(test_malignant, min(50, len(test_malignant)), replace=False)

    records = []
    
    # Map paths relative to data_dir and remove extension to perfectly fit the preprocessing loader
    for path, label in ([(p, 0) for p in selected_train_b] + 
                        [(p, 1) for p in selected_train_m] + 
                        [(p, 0) for p in selected_test_b] + 
                        [(p, 1) for p in selected_test_m]):
        p = Path(path)
        rel_path = p.relative_to(data_dir)
        rel_path_no_ext = str(rel_path.with_suffix(''))
        records.append({"image_id": rel_path_no_ext, "label": label})

    df = pd.DataFrame(records)
    df.to_csv(csv_path, index=False)
    print(f"\n[OK] Created metadata.csv with {len(df)} sampled real images.")
    print(f"     -> No images were duplicated or copied on disk!")
    return csv_path


# ─────────────────────────────────────────────────────────────────────────────
#  PHASE 2: FEATURE EXTRACTION — ResNet50 embeddings
# ─────────────────────────────────────────────────────────────────────────────
def phase2_feature_extraction(csv_path, force=False):
    print(SEP)
    print("  PHASE 2 — FEATURE EXTRACTION (ResNet50)")
    print(SEP)

    train_npz = paths.EMBEDDINGS_DIR / "train_embeddings.npz"
    val_npz   = paths.EMBEDDINGS_DIR / "val_embeddings.npz"
    test_npz  = paths.EMBEDDINGS_DIR / "test_embeddings.npz"

    if not force and train_npz.exists() and val_npz.exists() and test_npz.exists():
        print("[SKIP] All embedding files already exist.")
        return

    # 2a — DataLoaders
    print("\n[2a] Building DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(
        csv_file_path=str(csv_path),
        image_dir=str(paths.DATA_DIR)
    )
    if train_loader is None:
        print("[ERROR] Could not create DataLoaders. Check metadata.csv.")
        sys.exit(1)

    # 2b — Feature extractor
    print("\n[2b] Loading ResNet50 feature extractor (pretrained on ImageNet)...")
    feature_extractor, embedding_dim = get_feature_extractor("resnet50")
    print(f"      Output embedding dimension : {embedding_dim}")

    # 2c — Extract all splits
    for split_name, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
        print(f"\n[2c] Extracting {split_name.upper()} embeddings...")
        emb, lbl = extract_embeddings(loader, feature_extractor)
        save_embeddings(emb, lbl, split_name)

    print(f"\n[OK] Phase 2 done. Embeddings saved to: {paths.EMBEDDINGS_DIR}")


# ─────────────────────────────────────────────────────────────────────────────
#  PHASE 3: MULTI-MODEL TRAINING & EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
def phase3_training_and_evaluation():
    print(SEP)
    print("  PHASE 3 — MULTI-MODEL TRAINING & EVALUATION")
    print(SEP)

    # Ensure output dirs exist
    fig_dir = paths.REPORTS_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 3a — Load embeddings
    print("\n[3a] Loading embeddings from disk...")
    X_train, y_train = load_embeddings("train")
    X_val,   y_val   = load_embeddings("val")
    X_test,  y_test  = load_embeddings("test")
    print(f"      Train : {X_train.shape}  |  Val : {X_val.shape}  |  Test : {X_test.shape}")
    print(f"      Train class dist : Benign={np.sum(y_train==0)}, Melanoma={np.sum(y_train==1)}")

    # 3b — Define all classifiers to run
    MODEL_NAMES = ["svm", "logistic_regression", "random_forest", "adaboost", "lda"]

    results_summary = []   # will hold one dict per model
    best_model      = None
    best_auc        = -1.0
    best_name       = ""

    for model_name in MODEL_NAMES:
        print(SEP)
        print(f"  Training : {model_name.upper()}")
        print(SEP)

        clf = get_classifier(model_name, random_state=model_config.RANDOM_SEED)
        clf = train_classifier(clf, X_train, y_train)

        # Evaluate on validation set
        print(f"\n  >> VAL performance:")
        val_preds, val_probs = predict_classifier(clf, X_val)
        val_metrics = evaluate_classifier(y_val, val_preds, val_probs, is_multiclass=False)

        # Evaluate on test set
        print(f"\n  >> TEST performance:")
        test_preds, test_probs = predict_classifier(clf, X_test)
        test_metrics = evaluate_classifier(y_test, test_preds, test_probs, is_multiclass=False)

        # Save confusion matrix for each model
        plot_confusion_matrix(
            y_test, test_preds,
            class_names=["Benign", "Melanoma"],
            filename=f"confusion_matrix_{model_name}.png"
        )

        # Save ROC curve for each model (if probabilities available)
        if test_probs is not None:
            plot_roc_curve(y_test, test_probs, filename=f"roc_curve_{model_name}.png")

        auc = test_metrics.get("AUC-ROC", 0.0)
        if isinstance(auc, float) and auc > best_auc:
            best_auc   = auc
            best_model = clf
            best_name  = model_name

        results_summary.append({
            "Model"       : model_name.replace("_", " ").title(),
            "Val Acc"     : f"{val_metrics.get('Accuracy',0):.4f}",
            "Test Acc"    : f"{test_metrics.get('Accuracy',0):.4f}",
            "Test F1"     : f"{test_metrics.get('F1-Score',0):.4f}",
            "Test AUC"    : f"{test_metrics.get('AUC-ROC', 'N/A')}",
        })

    # ── 3c: Save comparison bar chart ─────────────────────────────────────────
    print(SEP)
    print("  SAVING COMPARISON CHART")
    print(SEP)

    df_results = pd.DataFrame(results_summary)
    _save_comparison_chart(df_results, fig_dir)

    # ── 3d: Save best model ───────────────────────────────────────────────────
    model_path = paths.MODELS_DIR / "best_classifier.joblib"
    joblib.dump(best_model, model_path)
    print(f"\n[OK] Best model ({best_name}, AUC={best_auc:.4f}) saved to: {model_path}")

    # ── 3e: Final summary table ───────────────────────────────────────────────
    print(SEP)
    print("  FINAL COMPARISON TABLE")
    print(SEP)
    print(df_results.to_string(index=False))
    print(f"\n[BEST] {best_name.upper()}  (Test AUC = {best_auc:.4f})")
    print(f"\n  Output plots   -> {fig_dir}")
    print(f"  Best model     -> {model_path}")
    print(SEP)

    return df_results


def _save_comparison_chart(df, fig_dir):
    """Saves a grouped bar chart comparing all models on Test Acc, F1, AUC."""
    models   = df["Model"].tolist()
    test_acc = [float(v) for v in df["Test Acc"]]
    test_f1  = [float(v) for v in df["Test F1"]]
    test_auc = []
    for v in df["Test AUC"]:
        try:
            test_auc.append(float(v))
        except (ValueError, TypeError):
            test_auc.append(0.0)

    x    = np.arange(len(models))
    w    = 0.25
    fig, ax = plt.subplots(figsize=(12, 6))

    bars1 = ax.bar(x - w,   test_acc, w, label="Test Accuracy", color="#4C72B0")
    bars2 = ax.bar(x,       test_f1,  w, label="Test F1-Score", color="#DD8452")
    bars3 = ax.bar(x + w,   test_auc, w, label="Test AUC-ROC",  color="#55A868")

    ax.set_xlabel("Classifier", fontsize=12)
    ax.set_ylabel("Score",      fontsize=12)
    ax.set_title("DermAssist — Multi-Model Performance Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Value labels on bars
    for bar in [*bars1, *bars2, *bars3]:
        h = bar.get_height()
        ax.annotate(f"{h:.2f}",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    out = fig_dir / "model_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Comparison chart saved -> {out}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # force=True regenerates data + embeddings even if files already exist
    csv_path = phase1_data_setup(force=True)
    phase2_feature_extraction(csv_path, force=True)
    phase3_training_and_evaluation()
