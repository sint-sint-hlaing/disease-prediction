# Disease Prediction Web Application

Advanced machine learning-based disease prediction system with interactive web interface.

## Features

- **Multi-Model Support**: Random Forest, Gradient Boosting, SVM, Naive Bayes, KNN
- **Interactive UI**: Real-time symptom selection with search functionality
- **Accuracy Visualization**: Comparative charts and confusion matrices
- **Top-N Predictions**: Shows top 3 disease predictions with confidence scores
- **Prediction History**: Tracks recent predictions
- **Model Comparison**: Visual comparison of all model performances
- **Responsive Design**: Bootstrap-based modern UI

## Project Structure

```
disease_prediction/
├── app.py                  # Flask application
├── train_models.py         # Model training script
├── create_dataset.py       # Dataset generation
├── requirements.txt        # Dependencies
├── models/                 # Trained models
├── data/                   # Dataset files
├── static/
│   ├── css/
│   │   └── style.css      # Styling
│   └── js/
│       └── app.js         # Frontend logic
└── templates/
    └── index.html         # Main page
```

## Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate Dataset
```bash
python create_dataset.py
```

### 3. Train Models
```bash
python train_models.py
```

### 4. Run Application
```bash
python app.py
```

### 5. Access Application
Open browser and navigate to: `http://localhost:5000`

## Usage

1. **Select Symptoms**: Check symptoms from the list or use search
2. **Choose Model**: Select ML algorithm from dropdown
3. **Predict**: Click "Predict Disease" button
4. **View Results**: See predicted disease with confidence score
5. **Compare Models**: View accuracy comparison charts

## Models Included

- **Random Forest**: Ensemble learning (Best accuracy)
- **Gradient Boosting**: Sequential ensemble method
- **SVM**: Support Vector Machine with RBF kernel
- **Naive Bayes**: Probabilistic classifier
- **KNN**: K-Nearest Neighbors

## Metrics Displayed

- Accuracy Score
- Cross-Validation Mean & Std
- Confusion Matrix
- Feature Importance
- Confidence Scores

## Technologies

- **Backend**: Flask, scikit-learn, pandas, numpy
- **Frontend**: HTML5, CSS3, JavaScript, Bootstrap 5
- **Visualization**: Chart.js, Matplotlib, Seaborn
- **ML**: Multiple classification algorithms

## API Endpoints

- `GET /` - Main page
- `POST /predict` - Disease prediction
- `GET /model_comparison` - Model metrics
- `GET /history` - Prediction history
- `GET /stats` - System statistics

## Future Enhancements

- Deep learning models (CNN, LSTM)
- User authentication
- Database integration (SQLite/PostgreSQL)
- Export reports (PDF)
- Mobile app version
- Real-time model retraining

## License

MIT License
