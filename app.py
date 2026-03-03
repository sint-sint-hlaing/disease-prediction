from flask import Flask, render_template, request, jsonify
import joblib
import json
import numpy as np
import pandas as pd
from datetime import datetime

app = Flask(__name__)

# Load models and metadata
with open('models/metadata.json', 'r') as f:
    metadata = json.load(f)

with open('models/model_results.json', 'r') as f:
    model_results = json.load(f)

features = metadata.get('selected_features', metadata['features'])
diseases = metadata['diseases']

# Load all models
models = {}
model_names = ['random_forest', 'gradient_boosting', 'svm', 'naive_bayes', 'knn']
for model_name in model_names:
    try:
        models[model_name] = joblib.load(f'models/{model_name}.pkl')
    except:
        pass

# Prediction history
prediction_history = []

@app.route('/')
def index():
    return render_template('index.html', symptoms=features, diseases=diseases)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    selected_symptoms = data.get('symptoms', [])
    selected_model = data.get('model', 'random_forest')
    
    # Create feature vector
    feature_vector = [1 if symptom in selected_symptoms else 0 for symptom in features]
    feature_array = np.array(feature_vector).reshape(1, -1)
    
    # Get model
    model = models.get(selected_model, models['random_forest'])
    
    # Predict
    prediction = model.predict(feature_array)[0]
    probabilities = model.predict_proba(feature_array)[0]
    
    # Get top 3 predictions
    top_indices = np.argsort(probabilities)[-3:][::-1]
    top_predictions = [
        {
            'disease': model.classes_[idx],
            'probability': float(probabilities[idx])
        }
        for idx in top_indices
    ]
    
    # Save to history
    prediction_entry = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'symptoms': selected_symptoms,
        'prediction': prediction,
        'confidence': float(max(probabilities)),
        'model': selected_model
    }
    prediction_history.append(prediction_entry)
    
    return jsonify({
        'prediction': prediction,
        'confidence': float(max(probabilities)),
        'top_predictions': top_predictions,
        'model_used': selected_model
    })

@app.route('/model_comparison')
def model_comparison():
    return jsonify(model_results)

@app.route('/evaluation_metrics/<model_name>')
def evaluation_metrics(model_name):
    from sklearn.metrics import confusion_matrix
    
    # Load data
    df = pd.read_csv('data/disease_symptom_data.csv')
    X = df.drop('disease', axis=1)
    y = df['disease']
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Load model
    model = models.get(model_name)
    if not model:
        return jsonify({'error': 'Model not found'}), 404
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Calculate metrics per class
    classes = model.classes_ if hasattr(model, 'classes_') else diseases
    metrics = []
    
    for disease in classes:
        y_test_binary = (y_test == disease).astype(int)
        y_pred_binary = (y_pred == disease).astype(int)
        
        TP = int(np.sum((y_test_binary == 1) & (y_pred_binary == 1)))
        TN = int(np.sum((y_test_binary == 0) & (y_pred_binary == 0)))
        FP = int(np.sum((y_test_binary == 0) & (y_pred_binary == 1)))
        FN = int(np.sum((y_test_binary == 1) & (y_pred_binary == 0)))
        
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        
        metrics.append({
            'disease': disease,
            'TP': TP,
            'TN': TN,
            'FP': FP,
            'FN': FN,
            'precision': round(precision, 4),
            'recall': round(recall, 4)
        })
    
    return jsonify(metrics)

@app.route('/history')
def history():
    return jsonify(prediction_history[-10:])

@app.route('/stats')
def stats():
    return jsonify({
        'total_predictions': len(prediction_history),
        'total_symptoms': len(features),
        'total_diseases': len(diseases),
        'models_available': len(models)
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
