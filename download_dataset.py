from datasets import load_dataset

# טוען את הדאטה מ-HuggingFace
ds = load_dataset("nkazi/MohlerASAG", name="raw")

# שמירת הדאטה כ-CSV בתיקיית data
import pandas as pd
import os

os.makedirs("data", exist_ok=True)

for split_name, split_data in ds.items():
    df = split_data.to_pandas()
    df.to_csv(f"data/mohler_{split_name}.csv", index=False)

print("Done! Files saved in /data/")
