from src.data_loader import load_data
from src.preprocessing import (
    clean_data,
    split_features_target,
    identify_column_types,
    build_preprocessor,
    split_data
)
from src.eda import run_eda
from src.models import (
    build_logistic_regression,
    build_random_forest,
    build_random_forest_smote
)
from src.evaluate import evaluate_model, save_results

def main():
    # Step 1: Load raw data
    df = load_data()
    print("Raw data loaded successfully.")
    print(df.head())

    # Step 2: Clean data
    df = clean_data(df)
    print("\nData cleaned successfully.")

    # Step 3: EDA
    X, y = split_features_target(df)
    categorical_cols, numerical_cols = identify_column_types(X)
    run_eda(df, numerical_cols)
    print("\nEDA completed. Figures saved in outputs/figures.")

    # Step 4: Build preprocessing pipeline
    preprocessor = build_preprocessor(categorical_cols, numerical_cols)

    # Step 5: Split data
    X_train, X_test, y_train, y_test = split_data(X, y)
    print("\nTrain-test split completed.")

    # Step 6: Build models
    logistic_model = build_logistic_regression(preprocessor)
    rf_model = build_random_forest(preprocessor)
    rf_smote_model = build_random_forest_smote(preprocessor)

    # Step 7: Evaluate models
    results = []
    results.append(evaluate_model(
        logistic_model, X_train, X_test, y_train, y_test, "Logistic Regression"
    ))
    results.append(evaluate_model(
        rf_model, X_train, X_test, y_train, y_test, "Random Forest"
    ))
    results.append(evaluate_model(
        rf_smote_model, X_train, X_test, y_train, y_test, "Random Forest SMOTE"
    ))

    # Step 8: Save results
    comparison_df = save_results(results)
    print("\nFinal Model Comparison:")
    print(comparison_df)

if __name__ == "__main__":
    main()