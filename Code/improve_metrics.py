import pandas as pd

def augment_hard_negatives():
    # הוספת דוגמאות שנועדו לפתור שגיאות סמנטיות (כמו מילת שלילה שמשנה הכל)
    hard_negatives = [
        {"question": "What is a Stack?", "student_answer": "A data structure that is NOT LIFO.", "score": 0},
        {"question": "What is a Stack?", "student_answer": "It is exactly like a Queue.", "score": 0},
        {"question": "Explain O(1).", "student_answer": "It means the time grows linearly.", "score": 0}
    ]
    
    df_existing = pd.read_csv('processed_data.csv')
    df_aug = pd.concat([df_existing, pd.DataFrame(hard_negatives)], ignore_index=True)
    
    # כוונון פרמטרים נוסף (למשל Dropout) מתבצע בזמן האימון בקובץ הקודם
    df_aug.to_csv('processed_data_v2.csv', index=False)
    print("Improvements: Added hard negatives to processed_data_v2.csv")

if __name__ == "__main__":
    augment_hard_negatives()
