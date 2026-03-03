import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score
import joblib

# Load data
df = pd.read_csv('data/disease_symptom_data.csv')
X = df.drop('disease', axis=1)
y = df['disease']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Load model
model = joblib.load('models/naive_bayes.pkl')

# Predict
y_pred = model.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
print()

# For multi-class, calculate metrics per class
print("Classification Report:")
print(classification_report(y_test, y_pred))
print()

# Calculate TP, TN, FP, FN for each class
classes = model.classes_
print("Per-Class Metrics:")
print("-" * 60)
for i, disease in enumerate(classes):
    # Binary classification for this class vs all others
    y_test_binary = (y_test == disease).astype(int)
    y_pred_binary = (y_pred == disease).astype(int)
    
    TP = np.sum((y_test_binary == 1) & (y_pred_binary == 1))
    TN = np.sum((y_test_binary == 0) & (y_pred_binary == 0))
    FP = np.sum((y_test_binary == 0) & (y_pred_binary == 1))
    FN = np.sum((y_test_binary == 1) & (y_pred_binary == 0))
    
    print(f"\n{disease}:")
    print(f"  True Positive (TP):  {TP}")
    print(f"  True Negative (TN):  {TN}")
    print(f"  False Positive (FP): {FP}")
    print(f"  False Negative (FN): {FN}")
    
    if TP + FP > 0:
        precision = TP / (TP + FP)
        print(f"  Precision: {precision:.4f}")
    if TP + FN > 0:
        recall = TP / (TP + FN)
        print(f"  Recall: {recall:.4f}")
