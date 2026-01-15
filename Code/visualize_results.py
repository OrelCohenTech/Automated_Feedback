import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def create_visuals():
    # נתונים לדוגמה המבוססים על ניסויי האימון שלנו
    metrics = {
        'Model': ['SBERT (Baseline)', 'DeBERTa (Our Model)'],
        'Pearson Correlation': [0.11, 0.71],
        'RMSE': [0.45, 0.40]
    }
    df_metrics = pd.DataFrame(metrics)

    plt.figure(figsize=(10, 5))
    
    # גרף קורלציה
    plt.subplot(1, 2, 1)
    sns.barplot(x='Model', y='Pearson Correlation', data=df_metrics, palette='Blues')
    plt.title('Performance Comparison (Correlation)')
    plt.ylim(0, 1)

    # גרף טעות (RMSE)
    plt.subplot(1, 2, 2)
    sns.barplot(x='Model', y='RMSE', data=df_metrics, palette='Reds')
    plt.title('Error Rate (Lower is Better)')
    
    plt.tight_layout()
    plt.savefig('model_performance.png')
    print("Visuals: Charts saved as model_performance.png")

if __name__ == "__main__":
    create_visuals()
