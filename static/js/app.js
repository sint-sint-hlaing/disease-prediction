let modelComparisonChart = null;

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    loadModelComparison();
    loadStats();
    loadHistory();

   
    const modelSelect = document.getElementById('modelSelect');

    modelSelect.addEventListener('change', function() {
        const selectedModel = this.value; 
        console.log("Algorithm changed to:", selectedModel);
        
      
        loadEvaluationMetrics(selectedModel);
    });

    
    loadEvaluationMetrics(modelSelect.value);
    
    // Symptom search functionality
    document.getElementById('symptomSearch').addEventListener('input', function(e) {
        const searchTerm = e.target.value.toLowerCase();
        const symptoms = document.querySelectorAll('.symptom-item');
        
        symptoms.forEach(symptom => {
            const label = symptom.querySelector('label').textContent.toLowerCase();
            symptom.style.display = label.includes(searchTerm) ? 'block' : 'none';
        });
    });
    
    // Predict button
    document.getElementById('predictBtn').addEventListener('click', predictDisease);
    
    // Clear button
    document.getElementById('clearBtn').addEventListener('click', function() {
        document.querySelectorAll('.form-check-input').forEach(cb => cb.checked = false);
        document.getElementById('resultCard').style.display = 'none';
        updateSelectedSymptoms();
    });
    
    // Track symptom selection changes
    document.querySelectorAll('.form-check-input').forEach(cb => {
        cb.addEventListener('change', updateSelectedSymptoms);
    });
});

// Update selected symptoms display
function updateSelectedSymptoms() {
    const selectedSymptoms = [];
    const checkboxMap = {};
    
    document.querySelectorAll('.form-check-input:checked').forEach(cb => {
        const symptomName = cb.nextElementSibling.textContent.trim();
        selectedSymptoms.push(symptomName);
        checkboxMap[symptomName] = cb;
    });
    
    const container = document.getElementById('selectedSymptomsContainer');
    const list = document.getElementById('selectedSymptomsList');
    const count = document.getElementById('symptomCount');
    
    if (selectedSymptoms.length === 0) {
        container.style.display = 'none';
    } else {
        container.style.display = 'block';
        count.textContent = selectedSymptoms.length;
        list.innerHTML = selectedSymptoms.map(s => 
            `<span class="badge bg-primary me-1 mb-1" style="cursor: pointer;" data-symptom="${s}">${s} ×</span>`
        ).join('');
        
        // Add click handlers to remove symptoms
        list.querySelectorAll('.badge').forEach(badge => {
            badge.addEventListener('click', function() {
                const symptomName = this.dataset.symptom;
                if (checkboxMap[symptomName]) {
                    checkboxMap[symptomName].checked = false;
                    updateSelectedSymptoms();
                }
            });
        });
    }
}

// Load model comparison data
async function loadModelComparison() {
    try {
        const response = await fetch('/model_comparison');
        const data = await response.json();
        
        // Populate metrics table
        const tbody = document.querySelector('#metricsTable tbody');
        tbody.innerHTML = '';
        
        const models = Object.keys(data);
        models.forEach(model => {
            const row = tbody.insertRow();
            row.innerHTML = `
                <td><strong>${model}</strong></td>
                <td>${(data[model].accuracy * 100).toFixed(2)}%</td>
                <td>${(data[model].cv_mean * 100).toFixed(2)}%</td>
                <td>±${(data[model].cv_std * 100).toFixed(2)}%</td>
            `;
        });
        
        // Create chart
        const ctx = document.getElementById('modelComparisonChart').getContext('2d');
        
        if (modelComparisonChart) {
            modelComparisonChart.destroy();
        }
        
        modelComparisonChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: models,
                datasets: [{
                    label: 'Accuracy',
                    data: models.map(m => data[m].accuracy * 100),
                    backgroundColor: 'rgba(54, 162, 235, 0.6)',
                    borderColor: 'rgba(54, 162, 235, 1)',
                    borderWidth: 2
                }, {
                    label: 'CV Mean',
                    data: models.map(m => data[m].cv_mean * 100),
                    backgroundColor: 'rgba(75, 192, 192, 0.6)',
                    borderColor: 'rgba(75, 192, 192, 1)',
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        ticks: {
                            callback: function(value) {
                                return value + '%';
                            }
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: true,
                        position: 'top'
                    },
                    title: {
                        display: true,
                        text: 'Model Performance Metrics (%)'
                    }
                }
            }
        });
    } catch (error) {
        console.error('Error loading model comparison:', error);
    }
}

// Load statistics
async function loadStats() {
    try {
        const response = await fetch('/stats');
        const data = await response.json();
        
        document.getElementById('totalPredictions').textContent = data.total_predictions;
        document.getElementById('totalSymptoms').textContent = data.total_symptoms;
        document.getElementById('totalDiseases').textContent = data.total_diseases;
    } catch (error) {
        console.error('Error loading stats:', error);
    }
}

// Load prediction history
async function loadHistory() {
    try {
        const response = await fetch('/history');
        const data = await response.json();
        
        const historyList = document.getElementById('historyList');
        historyList.innerHTML = '';
        
        if (data.length === 0) {
            historyList.innerHTML = '<p class="text-muted">No predictions yet</p>';
            return;
        }
        
        data.reverse().forEach(item => {
            const div = document.createElement('div');
            div.className = 'history-item';
            div.innerHTML = `
                <div style="display: flex; justify-content: between; align-items: start; margin-bottom: 8px;">
                    <strong style="color: var(--text-dark); font-size: 15px;">${item.prediction}</strong>
                    <span style="color: var(--success); font-weight: 600; margin-left: auto;">${(item.confidence * 100).toFixed(1)}%</span>
                </div>
                <div style="font-size: 13px; color: var(--text-muted);">
                    <i class="fas fa-clock"></i> ${item.timestamp} | 
                    <i class="fas fa-microchip"></i> ${item.model.replace('_', ' ')} | 
                    <i class="fas fa-notes-medical"></i> ${item.symptoms.length} symptoms
                </div>
            `;
            historyList.appendChild(div);
        });
    } catch (error) {
        console.error('Error loading history:', error);
    }
}

// Predict disease
async function predictDisease() {
    const selectedSymptoms = [];
    document.querySelectorAll('.form-check-input:checked').forEach(cb => {
        selectedSymptoms.push(cb.value);
    });
    
    if (selectedSymptoms.length === 0) {
        alert('Please select at least one symptom');
        return;
    }
    
    const selectedModel = document.getElementById('modelSelect').value;
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                symptoms: selectedSymptoms,
                model: selectedModel
            })
        });
        
        const data = await response.json();
        
        // Display result
        document.getElementById('predictedDisease').textContent = data.prediction;
        document.getElementById('confidence').textContent = (data.confidence * 100).toFixed(2) + '%';
        document.getElementById('modelUsed').textContent = selectedModel.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase());
        
        // Display top predictions
        const topPredDiv = document.getElementById('topPredictions');
        topPredDiv.innerHTML = '';
        
        data.top_predictions.forEach((pred, index) => {
            const colors = ['success', 'info', 'warning'];
            const progressBar = `
                <div class="mb-2">
                    <div class="d-flex justify-content-between mb-2">
                        <span style="font-weight: 600; color: var(--text-dark);">${index + 1}. ${pred.disease}</span>
                        <span style="font-weight: 600; color: var(--primary);">${(pred.reliability_score * 100).toFixed(1)}%</span>
                    </div>
                    <div class="progress">
                        <div class="progress-bar bg-${colors[index]}" 
                             role="progressbar" 
                             style="width: ${pred.reliability_score * 100}%">
                            ${(pred.reliability_score * 100).toFixed(1)}%
                        </div>
                    </div>
                </div>
            `;
            topPredDiv.innerHTML += progressBar;
        });
        
        document.getElementById('resultCard').style.display = 'block';
        
        // Load disease metrics
        loadDiseaseMetrics(selectedModel, data.prediction);
        
        // Load evaluation metrics
        loadEvaluationMetrics(selectedModel);
        
        // Refresh stats and history
        loadStats();
        loadHistory();
        
    } catch (error) {
        console.error('Error predicting disease:', error);
        alert('Error making prediction. Please try again.');
    }
}

// Load disease metrics
async function loadDiseaseMetrics(modelName, diseaseName) {
    try {
        const response = await fetch(`/disease_metrics/${modelName}/${encodeURIComponent(diseaseName)}`);
        const data = await response.json();
        
        document.getElementById('metricsDiseaseName').textContent = diseaseName;
        document.getElementById('cmTP').textContent = data.confusion_matrix.TP;
        document.getElementById('cmTN').textContent = data.confusion_matrix.TN;
        document.getElementById('cmFP').textContent = data.confusion_matrix.FP;
        document.getElementById('cmFN').textContent = data.confusion_matrix.FN;
        document.getElementById('totalPositive').textContent = data.confusion_matrix.total_positive;
        document.getElementById('totalNegative').textContent = data.confusion_matrix.total_negative;
        
        document.getElementById('metricPrecision').textContent = (data.metrics.precision * 100).toFixed(2) + '%';
        document.getElementById('metricRecall').textContent = (data.metrics.recall * 100).toFixed(2) + '%';
        document.getElementById('metricF1').textContent = (data.metrics.f1_score * 100).toFixed(2) + '%';
        document.getElementById('metricSupport').textContent = data.metrics.support;
        
        document.getElementById('diseaseMetricsCard').style.display = 'block';
    } catch (error) {
        console.error('Error loading disease metrics:', error);
    }
}

// Load evaluation metrics
async function loadEvaluationMetrics(modelName) {
    try {
        const response = await fetch(`/evaluation_metrics/${modelName}`);
        const data = await response.json();
        
        const tbody = document.querySelector('#evaluationTable tbody');
        tbody.innerHTML = '';
        
        data.forEach(metric => {
            const row = tbody.insertRow();
            row.innerHTML = `
                <td><strong>${metric.disease}</strong></td>
                <td><span class="badge bg-success">${metric.TP}</span></td>
                <td><span class="badge bg-secondary">${metric.TN}</span></td>
                <td><span class="badge bg-warning">${metric.FP}</span></td>
                <td><span class="badge bg-danger">${metric.FN}</span></td>
                <td>${(metric.precision * 100).toFixed(2)}%</td>
                <td>${(metric.recall * 100).toFixed(2)}%</td>
            `;
        });
        
        document.getElementById('evaluationCard').style.display = 'block';
    } catch (error) {
        console.error('Error loading evaluation metrics:', error);
    }
}


