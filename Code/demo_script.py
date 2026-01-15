import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def setup():
    print("--- Step 1: Installing & Loading Libraries ---")
    # פקודה להתקנה (להרצה בטרמינל אם חסר)
    # pip install transformers torch sentencepiece
    
    # טעינת המודל והטוקנייזר (משתמשים ב-DeBERTa v3 small כפי שתכננו)
    model_name = "microsoft/deberta-v3-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # במידה ויש לכם מודל מאומן מקומית, ניתן להחליף את הנתיב לתיקייה שלכם
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
    model.eval()
    return tokenizer, model

def predict_score(question, reference, student_answer, tokenizer, model):
    # יצירת ה-Input בפורמט Triple-Context (שאלה + רפרנס + תשובה)
    input_text = f"Question: {question} [SEP] Reference: {reference} [SEP] Student: {student_answer}"
    
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = model(**inputs)
        # המודל מחזיר ערך רציף (Regression), אנחנו מנרמלים אותו לסולם 0-2
        score = outputs.logits.item()
        # הגבלה לטווח הגיוני
        final_score = max(0, min(2, score * 2)) 
    return final_score

def run_demo():
    tokenizer, model = setup()
    
    # 2-3 דוגמאות להדגמה
    examples = [
        {
            "q": "What is a Linked List?",
            "ref": "A linear data structure where elements are stored in nodes connected by pointers.",
            "ans": "It's a collection of nodes where each node points to the next one using a pointer."
        },
        {
            "q": "What is a Linked List?",
            "ref": "A linear data structure where elements are stored in nodes connected by pointers.",
            "ans": "A list of numbers stored in an array." # תשובה שגויה סמנטית
        },
        {
            "q": "Define a Stack.",
            "ref": "A data structure that follows the Last-In-First-Out (LIFO) principle.",
            "ans": "A way to store data where the last item added is the first one to be removed."
        }
    ]
    
    print("\n--- Step 2: Running Inference Examples ---")
    for i, ex in enumerate(examples):
        score = predict_score(ex['q'], ex['ref'], ex['ans'], tokenizer, model)
        print(f"\nExample {i+1}:")
        print(f"Question: {ex['q']}")
        print(f"Student Answer: {ex['ans']}")
        print(f"Predicted Score (0-2): {score:.2f}")
        
        # פרשנות קצרה לציון
        if score > 1.5: status = "Correct"
        elif score > 0.7: status = "Partial"
        else: status = "Incorrect"
        print(f"Status: {status}")

if __name__ == "__main__":
    run_demo()demo_script.py
