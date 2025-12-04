
# 1. Imports and setup
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from datasets import concatenate_datasets

# 2. Load the dataset
print("Loading dataset...")
ds = load_dataset("nkazi/MohlerASAG")
print(ds)

# take a few examples

example1 = ds["open_ended"][0]
example2 = ds["open_ended"][1]
example3 = ds["open_ended"][2]

ds_raw = load_dataset("nkazi/MohlerASAG", name="raw")  # or default if appropriate
combined = concatenate_datasets([ds_raw["open_ended"], ds_raw["close_ended"]])
print(len(combined))
print(combined[0])


# 3. Format examples

def make_input(example):
    return (
        "QUESTION: " + example["question"] + "\n"
        "CORRECT ANSWER: " + example["instructor_answer"] + "\n"
        "STUDENT ANSWER: " + example["student_answer"]
    )

inputs = [
    make_input(example1),
    make_input(example2),
    make_input(example3),
]


# 4. Load a HuggingFace model (example: DistilBERT)
print("Loading model...")
model_name = "distilbert-base-uncased-finetuned-sst-2-english"

classifier = pipeline(
    "text-classification",
    model=model_name,
    top_k=None
)


# 5. Inference on examples
print("\nRunning inference...")
for i, text in enumerate(inputs):
    print(f"\n--- Example {i+1} ---")
    result = classifier(text)
    print(result)
