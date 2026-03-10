import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import joblib

# Load data
print("Loading data...")
df = pd.read_csv('data/disease_symptom_data.csv')
X = df.drop('disease', axis=1)
y = df['disease']

# Split data (same as training)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Disease labels
disease_labels = sorted(y.unique())

# Models to generate confusion matrices
models = {
    'Gradient Boosting': 'gradient_boosting',
    'SVM': 'svm',
    'Naive Bayes': 'naive_bayes',
    'KNN': 'knn'
}

print(f"\nGenerating confusion matrices for {len(models)} models...\n")

for model_name, model_file in models.items():
    try:
        # Load model
        model = joblib.load(f'models/{model_file}.pkl')
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Generate confusion matrix
        cm = confusion_matrix(y_test, y_pred, labels=disease_labels)
        
        # Create figure
        plt.figure(figsize=(14, 12))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=disease_labels, 
                    yticklabels=disease_labels,
                    cbar_kws={'label': 'Count'},
                    linewidths=0.5,
                    linecolor='gray')
        
        plt.title(f'Confusion Matrix - {model_name}', fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('True Label', fontsize=12, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right', fontsize=9)
        plt.yticks(rotation=0, fontsize=9)
        plt.tight_layout()
        
        # Save
        output_file = f'static/confusion_matrix_{model_file}.png'
        plt.savefig(output_file, dpi=100, bbox_inches='tight')
        plt.close()
        
        print(f"[OK] {model_name:20s} -> {output_file}")
        
    except FileNotFoundError:
        print(f"[ERROR] {model_name:20s} -> Model file not found: models/{model_file}.pkl")
    except Exception as e:
        print(f"[ERROR] {model_name:20s} -> Error: {str(e)}")

print("\n[DONE] Confusion matrix generation complete!")
