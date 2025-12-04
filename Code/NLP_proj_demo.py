# NLP_proj_demo.py

from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util

# 1. Load dataset
print("Loading dataset...")
ds = load_dataset("nkazi/MohlerASAG")

# 2. Select the first 3 examples from 'open_ended'
examples = [ds["open_ended"][i] for i in range(3)]

# 3. Load embedding model
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# 4. Compute similarity for each example
for i, ex in enumerate(examples, 1):
    stu_emb = model.encode(ex["student_answer"], convert_to_tensor=True)
    inst_emb = model.encode(ex["instructor_answer"], convert_to_tensor=True)
    similarity = util.cos_sim(stu_emb, inst_emb)
    
    # 5. Print formatted results
    print(f"\n--- Example {i} ---")
    print("Question:", ex["question"])
    print("Instructor Answer:", ex["instructor_answer"])
    print("Student Answer:", ex["student_answer"])
    print(f"Similarity score: {similarity.item():.4f}")
