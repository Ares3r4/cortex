import yaml
import numpy as np
import torch
import mlflow
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import f1_score, accuracy_score
from cortex.data_toxic import load_toxic_data

class LogDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(
            texts, 
            truncation=True, 
            padding=True, 
            max_length=max_length
        )
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro")
    }

def train_v2():
    with open("../configs/base.yaml", "r") as f:
        cfg = yaml.safe_load(f)
        
    cfg["max_lines"] = 10000
    
    print(f"Loading data for V2...")
    (x_train, y_train), (x_val, y_val) = load_toxic_data(cfg)
    print(f"Train size: {len(x_train)} | Val size: {len(x_val)}")

    print("Loading DistilBERT tokenizer...")
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print("Tokenizing data...")
    train_dataset = LogDataset(x_train, y_train, tokenizer)
    val_dataset = LogDataset(x_val, y_val, tokenizer)

    print("Loading DistilBERT model...")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    mlflow.set_tracking_uri("sqlite:///../mlflow.db")
    mlflow.set_experiment("Cortex_V2_Toxic")
    
    with mlflow.start_run():
        print("Starting MLflow run...")
        
        training_args = TrainingArguments(
            output_dir="../artifacts/checkpoints",
            num_train_epochs=1,               
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            eval_strategy="epoch",            
            save_strategy="epoch",
            learning_rate=2e-5,
            report_to="none"                  
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
        )

        print("Training started...")
        trainer.train()

        print("Evaluating on validation set...")
        eval_results = trainer.evaluate()
        print("Results:", eval_results)
        
        mlflow.log_params({
            "model": model_name,
            "max_lines": cfg["max_lines"],
            "epochs": training_args.num_train_epochs,
            "batch_size": training_args.per_device_train_batch_size
        })
        mlflow.log_metric("val_macro_f1", eval_results["eval_macro_f1"])
        mlflow.log_metric("val_accuracy", eval_results["eval_accuracy"])

        print("Saving model to artifacts/model_v2...")
        trainer.save_model("../artifacts/model_v2")
        tokenizer.save_pretrained("../artifacts/model_v2")
        
        print("V2 Training Complete!")

if __name__ == "__main__":
    train_v2()