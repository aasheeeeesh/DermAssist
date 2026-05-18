import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)
from .config import paths

def evaluate_classifier(y_true, y_pred, y_prob=None, is_multiclass=False):
    """
    Calculates standard Machine Learning metrics.
    Works dynamically for both Binary and Multi-class setups.
    """
    print("\n--- Classification Report ---")
    print(classification_report(y_true, y_pred))
    
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
    }
    
    # Handle averaging dynamically depending on number of classes
    avg_type = "macro" if is_multiclass else "binary"
    
    try:
        metrics["Precision"] = precision_score(y_true, y_pred, average=avg_type)
        metrics["Recall"] = recall_score(y_true, y_pred, average=avg_type)
        metrics["F1-Score"] = f1_score(y_true, y_pred, average=avg_type)
        
        # Calculate AUC if probabilities are provided
        if y_prob is not None:
            if is_multiclass:
                metrics["AUC-ROC"] = roc_auc_score(y_true, y_prob, multi_class='ovr')
            else:
                # For binary, roc_auc_score expects probabilities of the positive class (usually index 1)
                metrics["AUC-ROC"] = roc_auc_score(y_true, y_prob[:, 1] if len(y_prob.shape) > 1 else y_prob)
    except Exception as e:
        print(f"Warning: Could not compute advanced metrics. Reason: {e}")
        
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
        
    return metrics

def plot_confusion_matrix(y_true, y_pred, class_names=None, filename="confusion_matrix.png"):
    """
    Draws a heatmap of the confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if class_names is None:
        class_names = [f"Class {i}" for i in range(cm.shape[0])]
        
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    save_path = paths.REPORTS_DIR / "figures" / filename
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved confusion matrix plot to {save_path}")

def plot_roc_curve(y_true, y_prob, filename="roc_curve.png"):
    """
    Plots the Receiver Operating Characteristic (ROC) curve.
    Currently assumes Binary Classification.
    """
    # Assuming positive class probabilities are in column 1
    if len(y_prob.shape) > 1:
        probs = y_prob[:, 1]
    else:
        probs = y_prob
        
    fpr, tpr, thresholds = roc_curve(y_true, probs)
    auc_score = roc_auc_score(y_true, probs)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    
    save_path = paths.REPORTS_DIR / "figures" / filename
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved ROC curve to {save_path}")
