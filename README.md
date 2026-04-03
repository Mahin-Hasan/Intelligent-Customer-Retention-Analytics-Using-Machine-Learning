# Intelligent Customer Retention Analytics Using Machine Learning

## Overview

This project predicts customer churn using machine learning techniques on the Telco Customer Churn dataset. The goal is to identify customers likely to leave and help businesses take proactive retention actions.

The project follows a structured pipeline:

* Week 1: Problem definition, dataset selection, environment setup
* Week 2: Data preprocessing and exploratory data analysis (EDA)
* Week 3: Model development, evaluation, and class imbalance handling

---

## Dataset

* Source: Telco Customer Churn Dataset (Kaggle)
* Records: ~7,000 customers
* Target Variable: `Churn` (Yes/No → 1/0)

Features include:

* Demographics (gender, senior citizen, etc.)
* Account details (tenure, contract type)
* Services (internet, tech support, etc.)
* Billing information (monthly charges, total charges)

---

## Requirements

```id="a1k9zs"
pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn
```

Install dependencies:

```id="u4m2vx"
pip install -r requirements.txt
```

---

## Run Project

```id="q7d3wp"
python -m src.main
```

