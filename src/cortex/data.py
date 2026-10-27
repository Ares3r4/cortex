from __future__ import annotations

import json
import random
import urllib.request
import zipfile
from pathlib import Path
from typing import Any
import numpy as np
from sklearn.model_selection import train_test_split

BGL_URL = "https://zenodo.org/records/8196385/files/BGL.zip?download=1"
BGL_ZIP = "BGL.zip"
BGL_LOG = "BGL.log"
LABELS = ["normal", "alert"]

def ensure_bgl(cfg: dict) -> Path:
    data_dir = Path(cfg["data_dir"])
    data_dir.mkdir(parents=True, exist_ok=True)
    log_path = data_dir / BGL_LOG

    if log_path.exists():
        return log_path

    if not cfg.get("auto_download", True):
        raise FileNotFoundError(
            f"{log_path} not found. Download BGL.zip from {BGL_URL} and unzip it here."
        )

    zip_path = data_dir / BGL_ZIP
    urllib.request.urlretrieve(BGL_URL, zip_path)

    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if name.endswith(BGL_LOG):
                with zf.open(name) as src, open(log_path, "wb") as dst:
                    dst.write(src.read())
                break

    zip_path.unlink(missing_ok=True)
    return log_path

def _parse_line(line: str) -> tuple[str, str] | None:
    parts = line.split(maxsplit=9)
    if len(parts) < 10:
        return None
    label_tok, message = parts[0], parts[9].strip()
    if not message:
        return None
    return message, ("normal" if label_tok == "-" else "alert")

def split_indices(n: int, y: "np.ndarray", val_size: float, test_size: float, seed:int) -> dict:
    idx = np.arange(n)
    tmp_size = val_size + test_size
    assert 0 < val_size and 0 < test_size and tmp_size < 1.0

    idx_train, idx_tmp, y_train, y_tmp = train_test_split(
        idx, y, test_size=tmp_size, stratify=y, random_state = seed
    )

    rel_test = test_size / tmp_size
    
    idx_val, idx_test, y_val, y_test = train_test_split(
        idx_tmp, y_tmp, test_size=rel_test, stratify=y_tmp, random_state=seed
    )
    
    return {"train": idx_train, "val": idx_val, "test": idx_test}

def persist_splits(splits: dict, y:"np.ndarray", cfg: dict) -> None:
    splits_dir = Path(cfg["splits_dir"])
    splits_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        splits_dir / "indices.npz",
        train=splits["train"],
        val=splits["val"],
        test=splits["test"]
    )

    meta = {
        "seed"      :  cfg["seed"],
        "val_size"  :  cfg["val_size"],
        "test_size" :  cfg["test_size"],
        "sizes": {k: len(v) for k, v in splits.items()},
        "class_counts": {
            split: np.bincount(y[ids], minlength=2).tolist()
            for split, ids in splits.items()
        }
    }

    with open(splits_dir / "split_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

def load_saved_split(cfg: dict, texts: list[str], y: "np.ndarray") -> dict:
    splits_dir= Path(cfg["splits_dir"])
    npz_path = splits_dir / "indices.npz"

    if not npz_path.exists():
        raise FileNotFoundError(f"Split indices not found at {npz_path}. Run the split generation first!"
        )
    npz = np.load(npz_path)
    
    return {
        "train": ([texts[i] for i in npz["train"]], y[npz["train"]]),
        "val":   ([texts[i] for i in npz["val"]],   y[npz["val"]]),
        "test":  ([texts[i] for i in npz["test"]],  y[npz["test"]]),
        "indices": {"train": npz["train"], "val": npz["val"], "test": npz["test"]}
    }
    
def parse_bgl(path: Path, max_lines: int | None, seed: int) -> tuple[list[str], list[str]]:
    texts = []
    labels = []
    
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            parsed = _parse_line(line)
            
            if parsed is None:
                continue
                
            msg, label = parsed
            texts.append(msg)
            labels.append(label)
            
            if max_lines is not None and len(texts) >= max_lines:
                break
                
    return texts, labels    

if __name__ == "__main__":
    import yaml
    
    with open("../configs/base.yaml", "r") as f:
        cfg = yaml.safe_load(f)
        
    print("Loading full dataset...")
    path = ensure_bgl(cfg)
    texts, raw_labels = parse_bgl(path, cfg.get("max_lines"), cfg["seed"])
    
    y = np.array([0 if label == "normal" else 1 for label in raw_labels])
    
    print(f"Total valid lines: {len(texts)}")
    
    print("Splitting indices...")
    splits = split_indices(len(texts), y, cfg["val_size"], cfg["test_size"], cfg["seed"])
    
    print("Saving splits...")
    persist_splits(splits, y, cfg)
    
    print("Done! You can now run train.py")