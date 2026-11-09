import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

def load_toxic_data(cfg):
    csv_path = Path("../data/raw/train.csv")
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {csv_path}. Download train.csv and put it there.")

    print("Loading CSV...")
    df = pd.read_csv(csv_path)
    
    # Combine the 6 toxicity columns into one label
    toxic_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    df['is_toxic'] = df[toxic_cols].max(axis=1)

    texts = df['comment_text'].astype(str).tolist()
    y = df['is_toxic'].values

    # Limit to 10,000 rows
    max_lines = cfg.get("max_lines", 10000)
    if max_lines is not None:
        texts = texts[:max_lines]
        y = y[:max_lines]

    print(f"Splitting {len(texts)} comments...")
    x_train, x_val, y_train, y_val = train_test_split(
        texts, y, test_size=0.15, stratify=y, random_state=cfg["seed"]
    )

    return (x_train, y_train), (x_val, y_val)