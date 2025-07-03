import os
import pandas as pd
from datetime import datetime

def register_model(model_path, metrics, registry_path='registry/model_registry.csv'):
    os.makedirs(os.path.dirname(registry_path), exist_ok=True)

    record = {
        'model_path': model_path,
        'registered_at': datetime.now().isoformat(),
        **metrics
    }

    if os.path.exists(registry_path):
        df = pd.read_csv(registry_path)
        df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
    else:
        df = pd.DataFrame([record])

    df.to_csv(registry_path, index=False)
    print(f"[✓] Model registered in {registry_path}")


def list_registered_models(registry_path='registry/model_registry.csv'):
    if os.path.exists(registry_path):
        return pd.read_csv(registry_path).to_dict(orient='records')
    return []
