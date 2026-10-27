import yaml
from pathlib import Path
import joblib
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from cortex.data import ensure_bgl, parse_bgl, load_saved_split

print("SCRIPT STARTED")

def train():
    # 1. Load config
    with open("../configs/base.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    print("Loading data...")
    path = ensure_bgl(cfg)
    texts, raw_labels = parse_bgl(path, cfg.get("max_lines"), cfg["seed"])
    
    y = np.array([0 if label == "normal" else 1 for label in raw_labels])

    print("Loading splits...")
    splits = load_saved_split(cfg, texts, y)
    x_train, y_train = splits["train"]
    x_val, y_val = splits["val"]

    print(f"Train size: {len(x_train)} | Val size: {len(x_val)}")
        # --- DIAGNOSTIC BLOCK ---
    print("Class distribution in y_train:", np.bincount(y_train))
    print("Class distribution in y_val:", np.bincount(y_val))
    
    train_set = set(x_train)
    val_set = set(x_val)
    overlap = train_set.intersection(val_set)
    print(f"Exact duplicate text strings between train and val: {len(overlap)}")
    # ------------------------

    print("/Baseline")

    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(x_train, y_train)
    y_pred_dummy = dummy.predict(x_val)

    dummy_f1 = f1_score(y_val, y_pred_dummy, average="macro")
    ''' print("Class distribution in y_val:", np.bincount(y_val)) '''
    print(f"Dummy Macro F1 : {dummy_f1:.3f}")

    print("Real model")

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True, 
            ngram_range=(1, 2), 
            max_features=100_000, 
            sublinear_tf=True
        )),
        ("clf", LogisticRegression(
            class_weight="balanced", 
            max_iter=1000, 
            random_state=cfg["seed"]
        ))
    ])

    print("Training the real model (this gonna take ages bro you're cooked)")
    pipeline.fit(x_train, y_train)

    y_pred_real = pipeline.predict(x_val)
    real_f1 = f1_score(y_val, y_pred_real, average="macro")
    print(f"Real Model Macro F1: {real_f1:.3f}")


    artifacts_dir = Path("../artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    model_path = artifacts_dir / "model_v1.joblib"
    joblib.dump(pipeline, model_path)
    
    print(f"\nModel saved successfully to {model_path}")

if __name__ == "__main__":
    train()
