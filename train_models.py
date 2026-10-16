import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from ml_engine import (
    FEATURES,
    build_models,
    generate_demo_dataset,
    evaluate_models,
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

def main():
    data_path = DATA_DIR / "careercompass_demo.csv"
    if data_path.exists():
        data = pd.read_csv(data_path)
    else:
        data = generate_demo_dataset(1200, seed=42)
        data.to_csv(data_path, index=False)

    models = build_models()
    trained, metrics = evaluate_models(data)

    for name, model in trained.items():
        safe = name.lower().replace(" ", "_").replace("/", "_")
        joblib.dump(model, MODEL_DIR / f"{safe}.joblib")

    with open(MODEL_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(MODEL_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump({"features": FEATURES, "best_model": max(metrics, key=lambda x: metrics[x]["f1"])}, f, indent=2)

    print("CareerCompass models trained successfully.")
    print(pd.DataFrame(metrics).T.sort_values("f1", ascending=False))

if __name__ == "__main__":
    main()
