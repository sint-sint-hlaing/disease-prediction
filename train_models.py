import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize
from sklearn.feature_selection import SelectKBest, chi2
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json

# Load data
df = pd.read_csv('data/disease_symptom_data.csv')

# Prepare features and target
X = df.drop('disease', axis=1)
y = df['disease']

print(f"Original features: {X.shape[1]}")

# Feature Selection using Chi-Square Test
print("\n=== Feature Selection (Chi-Square Test) ===")
k_features = min(100, X.shape[1])
selector = SelectKBest(chi2, k=k_features)
X_selected = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()].tolist()
print(f"Selected Top {k_features} features")

# Compare: Train with ALL features
print("\n=== Training with ALL features ===")
X_train_all, X_test_all, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model_all = RandomForestClassifier(n_estimators=100, random_state=42)
model_all.fit(X_train_all, y_train)
y_pred_all = model_all.predict(X_test_all)
acc_all = accuracy_score(y_test, y_pred_all)
print(f"Accuracy with ALL features: {acc_all:.4f}")

# Compare: Train with TOP 35 features
print(f"\n=== Training with TOP {k_features} features ===")
X_top100 = X[selected_features]
X_train, X_test, y_train, y_test = train_test_split(X_top100, y, test_size=0.2, random_state=42, stratify=y)

# Hyperparameter Tuning for Random Forest
print("\n=== Hyperparameter Tuning (Grid Search) ===")
print("Training Default Random Forest...")
rf_default = RandomForestClassifier(random_state=42)
rf_default.fit(X_train, y_train)
y_pred_default = rf_default.predict(X_test)
acc_default = accuracy_score(y_test, y_pred_default)
print(f"Default RF Accuracy: {acc_default:.4f}")

print("\nPerforming Grid Search...")
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)
print(f"\nBest Parameters: {grid_search.best_params_}")

rf_tuned = grid_search.best_estimator_
y_pred_tuned = rf_tuned.predict(X_test)
acc_tuned = accuracy_score(y_test, y_pred_tuned)
print(f"Tuned RF Accuracy: {acc_tuned:.4f}")
print(f"\nImprovement: {acc_tuned - acc_default:.4f} ({((acc_tuned - acc_default) / acc_default * 100):.2f}%)")

# Train multiple models
models = {
    'Random Forest': rf_tuned,
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(kernel='rbf', probability=True, random_state=42),
    'Naive Bayes': GaussianNB(),
    'KNN': KNeighborsClassifier(n_neighbors=5)
}

results = {}
trained_models = {}

print("\n=== Training Other Models ===")
for name, model in models.items():
    if name == 'Random Forest':
        # Already trained
        results[name] = {
            'accuracy': float(acc_tuned),
            'cv_mean': float(cross_val_score(model, X_train, y_train, cv=5).mean()),
            'cv_std': float(cross_val_score(model, X_train, y_train, cv=5).std()),
            'classification_report': classification_report(y_test, y_pred_tuned, output_dict=True)
        }
        trained_models[name] = model
        print(f"Random Forest (Tuned) - Accuracy: {acc_tuned:.4f}")
        continue
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
    
    # Calculate ROC AUC
    if hasattr(model, 'predict_proba'):
        y_test_bin = label_binarize(y_test, classes=np.unique(y))
        y_pred_proba = model.predict_proba(X_test)
        roc_auc = roc_auc_score(y_test_bin, y_pred_proba, average='weighted', multi_class='ovr')
        results[name]['roc_auc'] = float(roc_auc)
        print(f"{name} - ROC AUC: {roc_auc:.4f}")

# Comparison Results
print("\n=== COMPARISON RESULTS ===")
print(f"\n1. Feature Selection:")
print(f"   ALL Features ({X.shape[1]}): Accuracy = {acc_all:.4f}")
print(f"   TOP {k_features} Features: Accuracy = {acc_tuned:.4f}")
print(f"\n2. Hyperparameter Tuning:")
print(f"   Default RF: Accuracy = {acc_default:.4f}")
print(f"   Tuned RF: Accuracy = {acc_tuned:.4f}")
print(f"   Improvement: {acc_tuned - acc_default:.4f} ({((acc_tuned - acc_default) / acc_default * 100):.2f}%)")
print(f"\n3. ROC AUC Scores:")
for name in trained_models.keys():
    if 'roc_auc' in results[name]:
        print(f"   {name}: AUC = {results[name]['roc_auc']:.4f}")

# Save best model (Random Forest)
best_model = trained_models['Random Forest']
joblib.dump(best_model, 'models/disease_predictor.pkl')

# Plot ROC Curves
print("\n=== Generating ROC Curves ===")
plt.figure(figsize=(10, 8))
y_test_bin = label_binarize(y_test, classes=np.unique(y))

for name, model in trained_models.items():
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test)
        
        # Compute micro-average ROC curve
        fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_pred_proba.ravel())
        roc_auc = roc_auc_score(y_test_bin, y_pred_proba, average='weighted', multi_class='ovr')
        
        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.3f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves - Multi-class Classification', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('static/roc_curve.png', dpi=100, bbox_inches='tight')
print("ROC curve saved to static/roc_curve.png")

# Save all models
for name, model in trained_models.items():
    safe_name = name.replace(' ', '_').lower()
    joblib.dump(model, f'models/{safe_name}.pkl')

# Save results
with open('models/model_results.json', 'w') as f:
    json.dump(results, f, indent=4)

# Save feature names and disease labels
feature_names = selected_features
disease_labels = y.unique().tolist()

with open('models/metadata.json', 'w') as f:
    json.dump({
        'features': feature_names,
        'diseases': disease_labels,
        'selected_features': selected_features,
        'feature_selection_method': 'chi2',
        'comparison': {
            'all_features': X.shape[1],
            'selected_features': k_features,
            'accuracy_all': float(acc_all),
            'accuracy_selected': float(acc_tuned),
            'rf_default_accuracy': float(acc_default),
            'rf_tuned_accuracy': float(acc_tuned),
            'best_params': grid_search.best_params_
        }
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

print("\n[SUCCESS] All models trained and saved successfully!")
print(f"[SUCCESS] Best model accuracy: {results['Random Forest']['accuracy']:.4f}")
