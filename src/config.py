import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
REPORTS_DIR = os.path.join(OUTPUT_DIR, "reports")

RANDOM_STATE = 42
TEST_SIZE = 0.20

TARGET_COLUMN = "Churn"

DROP_COLUMNS = ["customerID"]

NUMERICAL_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]