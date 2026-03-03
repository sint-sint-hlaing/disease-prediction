import pandas as pd
import numpy as np

# Comprehensive disease-symptom dataset
data = {
    'itching': [1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0],
    'skin_rash': [1,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
    'nodal_skin_eruptions': [1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
    'continuous_sneezing': [0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
    'shivering': [0,1,0,1,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0],
    'chills': [0,1,0,1,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0],
    'joint_pain': [0,0,1,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
    'stomach_pain': [0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
    'acidity': [0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
    'vomiting': [0,0,0,0,0,1,0,1,0,0,0,1,0,0,0,0,0,0,0,0],
    'fatigue': [0,1,1,1,0,0,1,0,0,0,1,0,0,1,0,0,0,0,0,0],
    'weight_loss': [0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0],
    'cough': [0,1,0,0,0,0,1,1,0,0,0,0,0,1,1,0,0,0,0,0],
    'high_fever': [0,1,0,1,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0],
    'breathlessness': [0,0,0,0,0,0,0,1,0,0,0,0,0,1,1,0,0,0,0,0],
    'sweating': [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],
    'headache': [0,1,0,1,0,0,0,0,0,1,0,0,0,0,0,1,1,0,0,0],
    'nausea': [0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,1,0,0,0],
    'loss_of_appetite': [0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,0,0,0,0,0],
    'muscle_pain': [0,1,1,1,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0],
    'chest_pain': [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0],
    'dizziness': [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0],
    'weakness': [0,0,1,1,0,0,1,1,0,0,1,0,0,1,0,0,1,0,0,0],
    'back_pain': [0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0],
    'constipation': [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0],
    'abdominal_pain': [0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,1,0],
    'diarrhoea': [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1],
    'mild_fever': [0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1],
    'yellow_urine': [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
    'yellowing_of_eyes': [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
    'swelling_of_stomach': [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
    'swollen_legs': [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
    'bloody_stool': [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],
    'irritability': [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0],
    'blurred_vision': [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0],
    'disease': ['Fungal infection','Common Cold','Arthritis','Malaria','Fungal infection',
                'GERD','Malaria','Tuberculosis','Fungal infection','Allergy','Arthritis',
                'Gastroenteritis','Hepatitis','Tuberculosis','Pneumonia','Dengue',
                'Migraine','Hypoglycemia','Peptic ulcer','Typhoid']
}

df = pd.DataFrame(data)

# Expand dataset with variations
expanded_data = []
for _ in range(50):
    for idx, row in df.iterrows():
        new_row = row.copy()
        # Add some noise
        symptom_cols = [col for col in df.columns if col != 'disease']
        for col in symptom_cols:
            if np.random.random() < 0.1:
                new_row[col] = 1 - new_row[col]
        expanded_data.append(new_row)

expanded_df = pd.DataFrame(expanded_data)
final_df = pd.concat([df, expanded_df], ignore_index=True)

final_df.to_csv('data/disease_symptom_data.csv', index=False)
print(f"Dataset created with {len(final_df)} records")
print(f"Diseases: {final_df['disease'].unique()}")
print(f"Symptoms: {len([col for col in final_df.columns if col != 'disease'])}")
