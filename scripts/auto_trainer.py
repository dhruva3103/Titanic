import pandas as pd
import time
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.model import train_and_evaluate

RAW_DATA_PATH = 'data/raw/incoming.csv'
MODEL_DIR = 'models/'
TARGET_COLUMN = 'Survived'

def get_next_version():
    registry_path = 'registry/model_registry.csv'
    if not os.path.exists(registry_path):
        return 1
    df = pd.read_csv(registry_path)
    return len(df) + 1

def auto_train():
    if not os.path.exists(RAW_DATA_PATH):
        print("No data yet...")
        return

    df = pd.read_csv(RAW_DATA_PATH)
    if df.shape[0] < 5:
        print("Not enough data yet...")
        return

    version = get_next_version()
    model_path = os.path.join(MODEL_DIR, f"model_v{version}.pkl")
    os.makedirs(MODEL_DIR, exist_ok=True)

    print(f"[✓] Training model version {version}...")
    train_and_evaluate(df, target_column=TARGET_COLUMN, output_model_path=model_path)

if __name__ == "__main__":
    while True:
        auto_train()
        time.sleep(60)  # wait 1 minute
