import pandas as pd
import numpy as np

def generate_enhanced_dataset():
    # יצירת דגימות בסיס כולל תשובות חלקיות ומטעות
    data = {
        'question': ["What is a Hash Table?", "Explain Big O notation.", "What is a Binary Tree?"] * 40,
        'reference': [
            "A data structure that maps keys to values using a hash function.",
            "A mathematical notation that describes the limiting behavior of a function.",
            "A tree data structure in which each node has at most two children."
        ] * 40,
        'student_answer': [
            "It maps keys to values with a function.", # 2 - Correct
            "It is a table with hashes for keys.",     # 1 - Partial
            "An array that stores data randomly.",    # 0 - Incorrect
            "Notation for algorithm complexity.",      # 2 - Correct
            "A way to measure time.",                  # 1 - Partial
            "O(n) is always the best case.",           # 0 - Incorrect (Hard Negative)
            "A tree where nodes have 0, 1, or 2 kids.",# 2 - Correct
            "A data structure with branches.",         # 1 - Partial
            "A linear list of nodes."                  # 0 - Incorrect
        ] * 13 + ["Final sample"] # השלמה ל-120 דוגמאות בערך
    }
    
    # חיתוך ויישור הנתונים
    df = pd.DataFrame(data).iloc[:120]
    
    # הוספת ציונים (Label)
    scores = [2, 1, 0, 2, 1, 0, 2, 1, 0] * 13 + [0, 0, 0]
    df['score'] = scores[:120]
    
    # Data Augmentation - הוספת רעש קל לטקסט לשיפור הרגישות
    df['student_answer'] = df['student_answer'].apply(lambda x: x + "." if np.random.rand() > 0.5 else x)
    
    df.to_csv('processed_data.csv', index=False)
    print("Pipeline: Dataset generated and saved to processed_data.csv")

if __name__ == "__main__":
    generate_enhanced_dataset()
