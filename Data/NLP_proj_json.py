import json
from datasets import load_dataset

# Load dataset
ds = load_dataset("nkazi/MohlerASAG")

# Take first 3 examples as a list of dicts
examples = [ds["open_ended"][i] for i in range(3)]

# Prepare data for JSON
data = []
for ex in examples:
    data.append({
        "id": ex["id"],
        "question": ex["question"],
        "instructor_answer": ex["instructor_answer"],
        "student_answer": ex["student_answer"],
        "score_avg": ex["score_avg"]
    })

# Save to JSON
json_file = "asag_examples.json"
with open(json_file, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"Saved {len(data)} examples to {json_file}")
