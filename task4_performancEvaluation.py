import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, precision_recall_curve, accuracy_score,
    precision_score, recall_score, f1_score
)
from sklearn.preprocessing import label_binarize

# === Load results from CSV ===
gcn_df = pd.read_csv("gcn_results.csv")
gnn_df = pd.read_csv("gnn_results.csv")

# === Define class names in order (excluding 'Normal-like') ===
class_names = ['Luminal A', 'Luminal B', 'HER2-enriched', 'Basal-like']

# === Remove rows with 'Normal-like' from both DataFrames ===
gcn_df = gcn_df[gcn_df['true_label'].isin(class_names)]
gnn_df = gnn_df[gnn_df['true_label'].isin(class_names)]

# === Map class labels to integers ===
label_to_int = {label: i for i, label in enumerate(class_names)}
y_true = gcn_df['true_label'].map(label_to_int).values
y_pred_gcn = gcn_df['predicted_label'].map(label_to_int).fillna(-1).astype(int).values
y_pred_sage = gnn_df['predicted_label'].map(label_to_int).fillna(-1).astype(int).values

# Remove predictions that are -1 (e.g., predicted as "Normal-like")
valid_idx_gcn = y_pred_gcn != -1
y_true_gcn = y_true[valid_idx_gcn]
y_pred_gcn = y_pred_gcn[valid_idx_gcn]

valid_idx_sage = y_pred_sage != -1
y_true_sage = y_true[valid_idx_sage]
y_pred_sage = y_pred_sage[valid_idx_sage]

# === Evaluation function ===
def evaluate_model(name, y_true, y_pred):
    print(f"\n=== {name} PERFORMANCE ===")
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    classification_rate = accuracy
    misclassification_rate = 1 - accuracy

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall (Sensitivity): {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Classification Rate: {classification_rate:.4f}")
    print(f"Misclassification Rate: {misclassification_rate:.4f}")
    
    # Only include present labels and their names
    unique_labels = np.unique(np.concatenate([y_true, y_pred]))
    label_names = [class_names[i] for i in unique_labels]
    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred, labels=unique_labels, target_names=label_names, zero_division=0))

    return {
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "Classification Rate": classification_rate,
        "Misclassification Rate": misclassification_rate
    }


# === Confusion Matrix Plot ===
def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'{title} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.show()

# === Precision-Recall Curves ===
def plot_precision_recall(y_true, y_pred, model_name):
    y_true_bin = label_binarize(y_true, classes=np.arange(len(class_names)))
    y_pred_bin = label_binarize(y_pred, classes=np.arange(len(class_names)))

    plt.figure(figsize=(7, 6))
    for i in range(len(class_names)):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_pred_bin[:, i])
        plt.plot(recall, precision, lw=2, label=f'{class_names[i]}')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{model_name} - Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# === Bar Plot of Metrics ===
def plot_metrics_bar(metrics_list):
    df = {
        "Metric": ["Accuracy", "Precision", "Recall", "F1-Score", "Classification Rate", "Misclassification Rate"]
    }
    for metrics in metrics_list:
        df[metrics["Model"]] = [
            metrics["Accuracy"], metrics["Precision"], metrics["Recall"],
            metrics["F1-Score"], metrics["Classification Rate"], metrics["Misclassification Rate"]
        ]
    df = pd.DataFrame(df)

    df.set_index("Metric").plot(kind="bar", figsize=(10, 6))
    plt.title("Model Performance Comparison")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.show()

# === Evaluate models ===
metrics_gcn = evaluate_model("GCN", y_true_gcn, y_pred_gcn)
metrics_sage = evaluate_model("GraphSAGE", y_true_sage, y_pred_sage)

# === Visualizations ===
plot_confusion_matrix(y_true_gcn, y_pred_gcn, title="GCN")
plot_confusion_matrix(y_true_sage, y_pred_sage, title="GraphSAGE")

plot_precision_recall(y_true_gcn, y_pred_gcn, model_name="GCN")
plot_precision_recall(y_true_sage, y_pred_sage, model_name="GraphSAGE")

plot_metrics_bar([metrics_gcn, metrics_sage])
