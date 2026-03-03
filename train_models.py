import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json

# Load data
df = pd.read_csv('data/disease_symptom_data.csv')

# Prepare features and target
X = df.drop('disease', axis=1)
y = df['disease']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train multiple models
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(kernel='rbf', probability=True, random_state=42),
    'Naive Bayes': GaussianNB(),
    'KNN': KNeighborsClassifier(n_neighbors=5)
}

results = {}
trained_models = {}

print("Training models...\n")
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    
    results[name] = {
        'accuracy': float(accuracy),
        'cv_mean': float(cv_scores.mean()),
        'cv_std': float(cv_scores.std()),
        'classification_report': classification_report(y_test, y_pred, output_dict=True)
    }
    trained_models[name] = model
    
    print(f"{name} - Accuracy: {accuracy:.4f}, CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# Save best model (Random Forest)
best_model = trained_models['Random Forest']
joblib.dump(best_model, 'models/disease_predictor.pkl')

# Save all models
for name, model in trained_models.items():
    safe_name = name.replace(' ', '_').lower()
    joblib.dump(model, f'models/{safe_name}.pkl')

# Save results
with open('models/model_results.json', 'w') as f:
    json.dump(results, f, indent=4)

# Save feature names and disease labels
feature_names = X.columns.tolist()
disease_labels = y.unique().tolist()

with open('models/metadata.json', 'w') as f:
    json.dump({
        'features': feature_names,
        'diseases': disease_labels
    }, f, indent=4)

# Generate visualizations
plt.figure(figsize=(12, 6))

# Accuracy comparison
plt.subplot(1, 2, 1)
accuracies = [results[name]['accuracy'] for name in models.keys()]
plt.bar(models.keys(), accuracies, color='skyblue')
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy')
plt.xticks(rotation=45, ha='right')
plt.ylim(0, 1)
for i, v in enumerate(accuracies):
    plt.text(i, v + 0.02, f'{v:.3f}', ha='center')

# Feature importance (Random Forest)
plt.subplot(1, 2, 2)
feature_importance = best_model.feature_importances_
top_features_idx = np.argsort(feature_importance)[-10:]
plt.barh([feature_names[i] for i in top_features_idx], feature_importance[top_features_idx])
plt.title('Top 10 Feature Importance (Random Forest)')
plt.xlabel('Importance')

plt.tight_layout()
plt.savefig('static/model_comparison.png', dpi=100, bbox_inches='tight')
print("\nVisualization saved to static/model_comparison.png")

# Confusion Matrix for best model
plt.figure(figsize=(10, 8))
y_pred_best = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=disease_labels, yticklabels=disease_labels)
plt.title('Confusion Matrix - Random Forest')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('static/confusion_matrix.png', dpi=100, bbox_inches='tight')
print("Confusion matrix saved to static/confusion_matrix.png")

print("\n✓ All models trained and saved successfully!")
print(f"✓ Best model accuracy: {results['Random Forest']['accuracy']:.4f}")
