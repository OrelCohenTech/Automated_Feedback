import csv
from datasets import load_dataset

# Load dataset
ds = load_dataset("nkazi/MohlerASAG")
examples = [ds["open_ended"][i] for i in range(3)]

# CSV file path
csv_file = "asag_examples.csv"

# Write CSV
with open(csv_file, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["id", "question", "instructor_answer", "student_answer", "score_avg"])
    writer.writeheader()
    for ex in examples:
        writer.writerow({
            "id": ex["id"],
            "question": ex["question"],
            "instructor_answer": ex["instructor_answer"],
            "student_answer": ex["student_answer"],
            "score_avg": ex["score_avg"]
        })

print(f"Saved {len(examples)} examples to {csv_file}")