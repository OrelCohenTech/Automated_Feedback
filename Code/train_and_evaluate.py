import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch

def train_model():
    df = pd.read_csv('processed_data.csv')
    dataset = Dataset.from_pandas(df)
    
    model_name = "microsoft/deberta-v3-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_func(examples):
        return tokenizer(examples['question'], examples['student_answer'], 
                         padding="max_length", truncation=True, max_length=128)

    tokenized_dataset = dataset.map(tokenize_func, batched=True)
    tokenized_dataset = tokenized_dataset.rename_column("score", "labels")
    tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.2)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)

    # שלב ד': Hyperparameter Tuning & Regularization מובנה כאן
    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=2e-5,          # Tuning
        per_device_train_batch_size=8,
        num_train_epochs=5,          # Iterative improvement
        weight_decay=0.01,           # Regularization
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
    )

    trainer.train()
    return trainer, tokenized_dataset["test"]

if __name__ == "__main__":
    trainer, test_data = train_model()
    results = trainer.evaluate()
    print(f"Evaluation Results: {results}")
