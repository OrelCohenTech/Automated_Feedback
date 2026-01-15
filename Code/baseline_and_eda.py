import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# 1. יצירת דגימת נתונים קטנה (Small Data Sample)
def generate_sample_data():
    data = {
        'Question': [
            "What is a Linked List?",
            "Explain the difference between Stack and Queue.",
            "What is the time complexity of searching in a Hash Table?",
            "What is a binary search tree?",
            "Define a recursive function."
        ],
        'Reference_Answer': [
            "A linear data structure where elements are stored in nodes and pointed by next pointers.",
            "A stack is LIFO (Last In First Out) and a queue is FIFO (First In First Out).",
            "The average time complexity is O(1).",
            "A tree where the left child is smaller and the right child is larger than the parent.",
            "A function that calls itself during its execution."
        ],
        'Student_Answer': [
            "It is a collection of nodes connected by pointers.", # Correct (2)
            "Stack is first in first out and queue is last in first out.", # Wrong Logic (0)
            "Usually it takes constant time, O(1).", # Correct (2)
            "A tree with nodes and branches.", # Partial (1)
            "A function that repeats using a loop.", # Wrong (0)
        ],
        'True_Score': [2, 0, 2, 1, 0]
    }
    return pd.DataFrame(data)

# 2. הרצת מודל "מדף" (Off-the-shelf SBERT)
def run_baseline(df):
    print("Running Baseline Model (SBERT)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # חישוב דמיון קוסינוס בין תשובת הסטודנט לתשובת הרפרנס
    ref_embeddings = model.encode(df['Reference_Answer'].tolist(), convert_to_tensor=True)
    stud_embeddings = model.encode(df['Student_Answer'].tolist(), convert_to_tensor=True)
    
    cosine_scores = util.cos_sim(ref_embeddings, stud_embeddings)
    
    # חילוץ הציון (האלכסון של המטריצה) ונרמול לסולם 0-2
    df['SBERT_Similarity'] = [cosine_scores[i][i].item() for i in range(len(df))]
    df['Predicted_Score'] = df['SBERT_Similarity'] * 2
    
    return df

# 3. ניתוח נתונים (EDA)
def perform_eda(df):
    print("\n--- Basic EDA ---")
    print(f"Dataset Size: {len(df)} samples")
    print(f"Average Student Answer Length: {df['Student_Answer'].apply(lambda x: len(x.split())).mean()} words")
    
    # חישוב קורלציה ראשונית
    corr, _ = pearsonr(df['True_Score'], df['Predicted_Score'])
    print(f"Initial Pearson Correlation: {corr:.4f}")

    # ויזואליזציה - התפלגות ציונים
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    sns.countplot(x='True_Score', data=df, palette='viridis')
    plt.title('Distribution of True Scores')

    plt.subplot(1, 2, 2)
    sns.scatterplot(x='True_Score', y='Predicted_Score', data=df)
    plt.title('True vs Predicted (Baseline)')
    
    plt.tight_layout()
    plt.savefig('baseline_eda_plots.png')
    print("EDA plots saved as 'baseline_eda_plots.png'")

# הרצה ראשית
if __name__ == "__main__":
    # יצירת דאטה
    df_sample = generate_sample_data()
    
    # הרצת מודל
    df_results = run_baseline(df_sample)
    
    # ניתוח תוצאות
    perform_eda(df_results)
    
    # הצגת התוצאות
    print("\n--- Final Table ---")
    print(df_results[['Student_Answer', 'True_Score', 'Predicted_Score']])
    
    # שמירה ל-CSV
    df_results.to_csv('baseline_results.csv', index=False)
