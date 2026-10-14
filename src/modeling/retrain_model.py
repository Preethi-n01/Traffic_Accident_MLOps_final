import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAINING_SCRIPT = PROJECT_ROOT / "src" / "modeling" / "train_xgboost.py"
MODEL_PATH = PROJECT_ROOT / "models" / "accident_severity_xgboost.joblib"
LOG_PATH = PROJECT_ROOT / "models" / "retraining_log.json"


def main():
    print("Starting XGBoost retraining workflow...")

    if not TRAINING_SCRIPT.exists():
        raise FileNotFoundError(
            f"Training script not found: {TRAINING_SCRIPT}"
        )

    result = subprocess.run(
        [sys.executable, str(TRAINING_SCRIPT)],
        cwd=PROJECT_ROOT,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "Model training failed. Check the training output."
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Expected model file was not found: {MODEL_PATH}"
        )

    if LOG_PATH.exists():
        with open(LOG_PATH, "r", encoding="utf-8") as file:
            log_data = json.load(file)
    else:
        log_data = {"runs": []}

    run_record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "training_script": "src/modeling/train_xgboost.py",
        "model_path": "models/accident_severity_xgboost.joblib",
        "status": "success",
        "note": (
            "Training rerun using the existing dataset and configuration. "
            "Review evaluation metrics before deployment."
        ),
    }

    log_data["runs"].append(run_record)

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(LOG_PATH, "w", encoding="utf-8") as file:
        json.dump(log_data, file, indent=4)

    print("\nRetraining completed successfully.")
    print(f"Model saved at: {MODEL_PATH}")
    print(f"Retraining log saved at: {LOG_PATH}")
    print("Evaluate the new model before using it for deployment.")


if __name__ == "__main__":
    main()