# Cortex (V1): Log Anomaly Detection

Cortex is an end-to-end machine learning baseline that classifies raw Blue Gene/L (BGL) system log messages as **Normal** or **Alert**. V1 emphasizes reproducible data splits, leakage prevention, and comparison against an honest majority-class baseline on a highly imbalanced dataset.

## Project Structure

```text
cortex/
├── configs/
│   └── base.yaml               # Dataset, split, and random-seed configuration
├── src/
│   ├── cortex/
│   │   ├── data.py             # BGL download, parsing, and split persistence
│   │   ├── evaluate.py         # Frozen-model test evaluation and error export
│   │   ├── features.py
│   │   ├── predict.py
│   │   ├── test_parse.py
│   │   └── train.py            # Dummy baseline and TF-IDF + logistic regression
│   └── data/                   # Dataset and split files created by the commands below
├── artifacts/
│   └── model_v1.joblib         # Serialized trained scikit-learn pipeline
├── reports/
│   ├── v1_metrics.json
│   └── v1_misclassified.csv
├── tests/
│   ├── test_data.py
│   └── test_train_smoke.py
├── Makefile
├── requirements.txt
└── README.md
```

## Dataset and Method

V1 uses the BGL supercomputer logs distributed by [LogHub](https://github.com/logpai/loghub). The dataset contains approximately 4.7 million log lines, with an approximate class distribution of 93% Normal and 7% Alert.

The parser extracts the message portion of each log line and removes its leading metadata before vectorization. This prevents the model from exploiting metadata that can directly encode the label. Data is divided into stratified train, validation, and test sets; the split indices are persisted in an `.npz` file so subsequent training and evaluation use the same examples.

The reference model is a scikit-learn `Pipeline` with a TF-IDF word unigram/bigram vectorizer and class-weighted logistic regression. A `DummyClassifier` using the most frequent class provides a deliberately simple baseline against which to judge performance.

## How to Run

The current scripts resolve configuration and data paths relative to the working directory. Run the Python modules from `src/` as shown below. The first data-preparation run downloads BGL automatically if it is not already present.

### Set up the environment

From the repository root, create and activate a virtual environment, then install the project dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On macOS or Linux, activate the environment with:

```bash
source .venv/bin/activate
```

### Prepare data, train, and evaluate

From the repository root, run:

```powershell
Set-Location src
python -m cortex.data
python -m cortex.train
python -m cortex.evaluate
```

`cortex.data` downloads and parses BGL, then writes stratified split indices and split metadata. `cortex.train` fits the baseline and TF-IDF + logistic regression model and saves the pipeline to `artifacts/model_v1.joblib`. `cortex.evaluate` loads that saved artifact and the locked test split, prints a classification report and confusion-matrix counts, and writes `reports/v1_metrics.json` and `reports/v1_misclassified.csv`.

The saved split indices are the source of truth for reproducibility: keep them with the dataset version when comparing model runs. If the dataset or split configuration changes, regenerate the splits deliberately before retraining and evaluating.

## Results & Metrics

The DummyClassifier score is the validation macro F1 reported during training. The learned model metrics below are from the locked test set (`n = 707,024`).

| Model / metric | Result |
|---|---:|
| Dummy baseline macro F1 (validation) | 0.481 |
| TF-IDF + logistic regression accuracy (test) | 99.99% |
| TF-IDF + logistic regression macro F1 (test) | 0.9997 |
| True Normals | 654,707 |
| False Alarms | 48 |
| Missed Alerts | 11 |
| True Alerts | 52,258 |

Despite the strong aggregate test metrics, accuracy alone is not sufficient for this imbalanced task. Macro F1 and the individual false-alarm and missed-alert counts provide a more informative view of minority-class performance and operational trade-offs.

## Error Analysis

Review of the misclassified logs highlights limitations that aggregate metrics do not capture:

- **Keyword overfitting:** The model flagged benign correction messages such as `tlb error....0` as alerts, indicating that it over-indexed on the phrase “tlb error.”
- **Semantic blindness:** It classified `ciod: Error creating node map...` as an alert based on the token “Error,” without recognizing the message as a benign startup log.
- **Rare alert blindness:** It missed 11 actual alerts, including messages such as `no ethernet link`. Extremely rare alert patterns are difficult to learn from the training distribution.

These errors motivate improvements in contextual language understanding and handling of rare classes; they also reinforce the need to inspect representative errors rather than relying on a single headline score.

## Next Steps (V2)

- Add **MLflow experiment tracking** for parameters, dataset/split metadata, metrics, and model artifacts to make comparisons auditable.
- Fine-tune a **DistilBERT** model on the log messages to capture context and address semantic blindness to isolated keywords.
- Evaluate rare-alert strategies and report per-class precision and recall alongside macro F1, with particular attention to missed alerts and false-alarm costs.
