import yaml
import json
import csv
import joblib
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report

from cortex.data import ensure_bgl, parse_bgl, load_saved_split

def evaluate():
    with open("../configs/base.yaml","r") as f:
        cfg=yaml.safe_load(f)

    print("loading data")
    path=ensure_bgl(cfg)
    text, raw_labels= parse_bgl(path, cfg.get("max_lines"), cfg["seed"])
    y = np.array([0 if label == "normal" else 1 for label in raw_labels])
    splits=load_saved_split(cfg,text,y)
    x_test, y_test=splits["test"]
    print(f"Test size: {len(x_test)} | Alerts in test: {np.bincount(y_test)[1]}")
    model_path= Path("../artifacts/model_v1.joblib")
    if not model_path.exists():
        raise FileNotFoundError(f"model not found at ${model_path}. Run train.py before first")
    print("Loading model")
    model =joblib.load(model_path)

    print("making prediction")
    y_pred=model.predict(x_test)

    print(classification_report(y_test, y_pred, target_names=["Normal", "Alert"]))
    print("confusion matrix")

    cm=confusion_matrix(y_test, y_pred)
    print(f"True Normals (Predicted Normal): {cm[0][0]}")
    print(f"False Alarms   (Predicted Alert): {cm[0][1]}")
    print(f"Missed Alerts  (Predicted Normal): {cm[1][0]}")
    print(f"True Alerts    (Predicted Alert): {cm[1][1]}")

    macro_f1=f1_score(y_test, y_pred,average="macro")
    acc=accuracy_score(y_test, y_pred)

    report = {
        "test_size": len(x_test),
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "confusion_matrix": {
            "true_normals": int(cm[0][0]),
            "false_alarms": int(cm[0][1]),
            "missed_alerts": int(cm[1][0]),
            "true_alerts": int(cm[1][1])
        }
    }
    reports_dir=Path("../reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    with open(reports_dir / "v1_metrics.json", "w") as f:
        json.dump(report, f, indent=2)
    print("Extracting misclassified logs...")
    misclassified_path = reports_dir / "v1_misclassified.csv"
    
    with open(misclassified_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "true_label", "pred_label"])
        
        mistakes = 0
        for text, true_y, pred_y in zip(x_test, y_test, y_pred):
            if true_y != pred_y:
                writer.writerow([text, true_y, pred_y])
                mistakes += 1
                
    print(f"Saved {mistakes} misclassified logs to {misclassified_path}")



if __name__ == "__main__":
    evaluate()