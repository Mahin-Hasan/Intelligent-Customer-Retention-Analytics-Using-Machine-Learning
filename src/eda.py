import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from src.config import FIGURES_DIR

os.makedirs(FIGURES_DIR, exist_ok=True)

sns.set_style("whitegrid")


def plot_churn_distribution(df):
    plt.figure(figsize=(6, 4))
    sns.countplot(x="Churn", data=df)
    plt.title("Churn Distribution")
    plt.xlabel("Churn")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "churn_distribution.png"))
    plt.close()


def plot_numerical_distributions(df, numerical_cols):
    for col in numerical_cols:
        plt.figure(figsize=(6, 4))
        sns.histplot(df[col], kde=True, bins=30)
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, f"{col}_histogram.png"))
        plt.close()

        plt.figure(figsize=(6, 4))
        sns.boxplot(x=df[col])
        plt.title(f"Boxplot of {col}")
        plt.xlabel(col)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, f"{col}_boxplot.png"))
        plt.close()


def plot_correlation_heatmap(df, numerical_cols):
    plt.figure(figsize=(8, 6))
    corr = df[numerical_cols].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "correlation_heatmap.png"))
    plt.close()


def plot_categorical_churn(df, column_name):
    churn_rate = pd.crosstab(df[column_name], df["Churn"], normalize="index")

    ax = churn_rate.plot(kind="bar", stacked=True, figsize=(8, 5))
    ax.set_title(f"Churn Rate by {column_name}")
    ax.set_xlabel(column_name)
    ax.set_ylabel("Proportion")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f"churn_by_{column_name}.png"))
    plt.close()


def run_eda(df, numerical_cols):
    plot_churn_distribution(df)
    plot_numerical_distributions(df, numerical_cols)
    plot_correlation_heatmap(df, numerical_cols)

    important_cat_cols = ["Contract", "InternetService", "TechSupport"]
    for col in important_cat_cols:
        if col in df.columns:
            plot_categorical_churn(df, col)