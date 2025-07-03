# scripts/simulate_stream.py
import pandas as pd
import time
import os

source = pd.read_csv("train.csv")
dest_path = "data/raw/incoming.csv"
os.makedirs("data/raw", exist_ok=True)

for i, row in source.iterrows():
    df = pd.DataFrame([row])
    if os.path.exists(dest_path):
        df.to_csv(dest_path, mode='a', header=False, index=False)
    else:
        df.to_csv(dest_path, index=False)
    print(f"[+] Added row {i+1}")
    time.sleep(60)  # wait 1 minute
