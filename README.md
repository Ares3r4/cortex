# Cortex

Cortex is a small machine-learning project with a FastAPI service for classifying
comments as **Toxic** or **Safe**. The API loads a fine-tuned DistilBERT model
from `artifacts/model_v2` and exposes prediction and health-check endpoints.

The repository also contains an earlier, independent experiment for classifying
Blue Gene/L (BGL) system logs as normal or alert. See
[the BGL experiment](#bgl-log-classification-legacy) for its commands.

## Contents

- [Requirements](#requirements)
- [Run the API locally](#run-the-api-locally)
- [Use the API](#use-the-api)
- [Run with Docker](#run-with-docker)
- [Train the toxic-comment model](#train-the-toxic-comment-model)
- [BGL log classification (legacy)](#bgl-log-classification-legacy)
- [Project layout](#project-layout)

## Requirements

- Python 3.12 (the version used by the Docker image)
- The dependencies in `requirements.txt`
- The model files in `artifacts/model_v2` to run the API

The model is loaded when the API starts. If the model directory is missing or
incompatible, startup will fail rather than serving predictions without a model.

## Run the API locally

From the repository root, create a virtual environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On macOS or Linux, activate the environment with:

```bash
source .venv/bin/activate
```

Start the server from the `src` directory. The working directory matters: the
API currently loads its model from `../artifacts/model_v2`.

```powershell
Set-Location src
python -m uvicorn cortex.api:app --host 127.0.0.1 --port 8000
```

The service is available at `http://127.0.0.1:8000`. Interactive API
documentation is at `http://127.0.0.1:8000/docs`.

## Use the API

Check whether the service is responding:

```powershell
curl.exe http://127.0.0.1:8000/health
```

Classify a comment:

```powershell
curl.exe -X POST http://127.0.0.1:8000/predict `
  -H "Content-Type: application/json" `
  -d '{"text":"Thanks for sharing your perspective."}'
```

`/predict` accepts a JSON object with a `text` string and returns the original
text, a Boolean `is_toxic`, a confidence score, and a `label` (`"Toxic"` or
`"Safe"`). For example:

```json
{
  "text": "Thanks for sharing your perspective.",
  "is_toxic": false,
  "confidence": 0.9876,
  "label": "Safe"
}
```

The confidence is the model's probability for its selected class; it is not a
calibrated measure of certainty.

## Run with Docker

Build the image from the repository root:

```powershell
docker build -t cortex-api .
```

Run the API and publish port 8000:

```powershell
docker run --rm -p 8000:8000 cortex-api
```

The image installs Python dependencies and copies `src/`, `configs/`, and
`artifacts/model_v2/`. It therefore requires the model directory to be present
at build time. Once running, use the same endpoints described above, for example
`http://localhost:8000/health`.

## Train the toxic-comment model

The training script expects a CSV at `data/raw/train.csv` with a `comment_text`
column and the six toxicity-label columns:
`toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, and `identity_hate`.
The labels are combined into a binary target: a comment is toxic if any of the
six labels is set.

From the repository root, run training from `src`:

```powershell
Set-Location src
python -m cortex.train_v2
```

On its first run, training downloads the `distilbert-base-uncased` tokenizer and
base model from Hugging Face, so network access is required. The current script
trains on the first 2,000 rows, uses a stratified 85/15 train/validation split,
runs for one epoch, and logs accuracy and macro F1 to MLflow. It saves the
fine-tuned model and tokenizer to `artifacts/model_v2` and checkpoints to
`artifacts/checkpoints`.

The API uses the saved model directory, so retrain before building the Docker
image if you want the image to include the newly trained model.

## BGL log classification (legacy)

The repository also includes a separate TF-IDF and logistic-regression baseline
for BGL system logs. It predicts **Normal** or **Alert** and is not used by the
FastAPI service.

From the repository root, generate data splits, train, and evaluate:

```powershell
Set-Location src
python -m cortex.data
python -m cortex.train
python -m cortex.evaluate
```

The first data command downloads BGL logs when they are not already present.
The split configuration is in `configs/base.yaml`; split indices and metadata
are written to `data/splits/`. Training saves `artifacts/model_v1.joblib`.
Evaluation uses the saved test split and writes metrics and misclassified
examples to `reports/`.

## Project layout

```text
cortex/
├── artifacts/
│   ├── model_v1.joblib          # Legacy BGL classifier
│   ├── model_v2/                # Fine-tuned toxic-comment model and tokenizer
│   └── checkpoints/             # Training checkpoints
├── configs/
│   └── base.yaml                # Shared data and split settings
├── data/
│   ├── raw/                     # Training CSV and downloaded BGL logs
│   └── splits/                  # Persisted BGL split indices and metadata
├── reports/                     # Legacy BGL evaluation outputs
├── src/cortex/
│   ├── api.py                   # FastAPI prediction service
│   ├── data.py                  # BGL download, parsing, and splits
│   ├── data_toxic.py            # Toxic-comment CSV loading
│   ├── evaluate.py              # Legacy BGL evaluation
│   ├── train.py                 # Legacy BGL baseline training
│   └── train_v2.py              # DistilBERT toxic-comment training
├── tests/
├── Dockerfile
└── requirements.txt
```
