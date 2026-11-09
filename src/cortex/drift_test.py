import requests
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset
from pathlib import Path

API_URL = "http://127.0.0.1:8000/predict"

# The 2026 Brainrot
test_texts = [
    "Bro really hit the griddy in Ohio, that shit is bussin no cap.",
    "I have immense rizz, my aura is literally glowing right now.",
    "Fanum tax took my lunch, but the gyatt was worth it.",
    "Your rizz is negative, you absolute beta NPC. You are cooked.",
    "Bro is posting L takes, go back to Ohio you absolute goofy ahh.",
    "This guy has zero aura. Truly a skill issue. Massive L.",
    "Imagine getting ratioed by a bot, couldn't be me. Cope and seethe.",
]

print("1. Sending texts to API (testing predictions & Prometheus metrics)...")
for text in test_texts:
    payload = {"text": text}
    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        data = response.json()
        print(f"   Text: {text[:30]}... -> Prediction: {data['label']} ({data['confidence']})")
    except requests.exceptions.ConnectionError:
        print("   ERROR: Is your Docker container running?")
        exit(1)

print("\n2. Generating Evidently Drift Report...")
# Load Reference Data (Training Data)
csv_path = Path("../data/raw/train.csv")
if not csv_path.exists():
    print("ERROR: Cannot find train.csv")
    exit(1)
    
df_train = pd.read_csv(csv_path, nrows=500)
reference_data = pd.DataFrame({"text": df_train['comment_text'].astype(str)})

# Load Current Data (The Brainrot we just sent)
current_data = pd.DataFrame({"text": test_texts})

# Run Evidently and get the result object
report = Report(metrics=[DataDriftPreset()])
result = report.run(reference_data=reference_data, current_data=current_data)

# Save to HTML using the result object
report_path = Path("../drift_report.html")
result.save_html(str(report_path))
print(f"\n✅ Drift report successfully saved to: {report_path.absolute()}")
print("Double-click that file to open the dashboard in your browser!")