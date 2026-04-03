import pandas as pd
from src.config import DATA_PATH

def load_data():
    try:
        df = pd.read_csv(DATA_PATH)
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset not found at path: {DATA_PATH}")
    except Exception as e:
        raise Exception(f"Error loading dataset: {e}")