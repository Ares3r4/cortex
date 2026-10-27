import numpy as np
from cortex.data import split_indices, persist_splits

def test_split():
    # 1. Fake a dataset of 10,000 rows. 93% normal (0), 7% alert (1)
    n = 10000
    rng = np.random.default_rng(0)
    y = rng.choice([0, 1], size=n, p=[0.93, 0.07])
    
    # 2. Run our function
    splits = split_indices(n, y, val_size=0.15, test_size=0.15, seed=42)
    
    # 3. Print the results
    sizes = {k: len(v) for k, v in splits.items()}
    print("Sizes:", sizes)
    
    for name, ids in splits.items():
        counts = np.bincount(y[ids], minlength=2)
        ratio = counts[1] / counts.sum()
        print(f"{name}: alerts={counts[1]}, ratio={ratio:.3f}")

def test_split():
    n = 10000
    rng = np.random.default_rng(0)
    y = rng.choice([0, 1], size=n, p=[0.93, 0.07])
    
    cfg = {
        "seed": 42,
        "val_size": 0.15,
        "test_size": 0.15,
        "splits_dir": "data/splits" # <--- Notice this path
    }
    
    splits = split_indices(n, y, val_size=cfg["val_size"], test_size=cfg["test_size"], seed=cfg["seed"])
    persist_splits(splits, y, cfg)
    print("Saved splits to disk!")

if __name__ == "__main__":
    test_split()