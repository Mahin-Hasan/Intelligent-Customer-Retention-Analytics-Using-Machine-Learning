import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score
)

from src.config import FIGURES_DIR, REPORTS_DIR

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    y_prob = None
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1-Score": f1_score(y_test, y_pred, zero_division=0)
    }

    if y_prob is not None:
        metrics["ROC-AUC"] = roc_auc_score(y_test, y_prob)

    print(f"\n===== {model_name} =====")
    for key, value in metrics.items():
        if key != "Model":
            print(f"{key}: {value:.4f}")

    report = classification_report(y_test, y_pred, zero_division=0)
    print("\nClassification Report:")
    print(report)

    safe_model_name = model_name.lower().replace(" ", "_").replace("-", "_")

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Churn", "Churn"],
        yticklabels=["No Churn", "Churn"]
    )
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f"{safe_model_name}_cm.png"))
    plt.close()

    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)

        plt.figure(figsize=(6, 4))
        plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.title(f"ROC Curve - {model_name}")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, f"{safe_model_name}_roc.png"))
        plt.close()

    report_path = os.path.join(REPORTS_DIR, f"{safe_model_name}_classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as file:
        file.write(f"{model_name}\n")
        file.write("=" * len(model_name) + "\n\n")
        file.write(report)

    return metrics


def save_results(results):
    df_results = pd.DataFrame(results)
    output_file = os.path.join(REPORTS_DIR, "model_results.csv")
    df_results.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    return df_results