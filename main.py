import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# NHS Diagnostic Assistant with Enhanced Features

# Expanded dataset with more symptoms and conditions (simulating a Kaggle-like dataset)
data = {
    'itching': [1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,0,1,0,1,0,1,0,1,0,1],
    'skin_rash': [1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,0,1,0,1,0,1,0,1,0],
    'continuous_sneezing': [0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,1,0,1,0,1,0,1,0,1,0],
    'shivering': [1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,0,1,0,1,0,1,0,1,0,1],
    'chills': [1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,0,1,0,1,0,1,0,1,0],
    'joint_pain': [1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,0,1,0,1,0,1,0,1,0,1],
    'stomach_pain': [0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,1,0,1,0,1,0,1,0,1,0],
    'acidity': [1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,0,1,0,1,0,1,0,1,0,1],
    'vomiting': [0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,1,0,1,0,1,0,1,0,1,0],
    'fatigue': [1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,0,1,0,1,0,1,0,1,0,1],
    'condition': ['Fungal infection'] * 5 + ['Allergy'] * 5 + ['GERD'] * 5 + ['Chronic cholestasis'] * 5 + ['Drug Reaction'] * 5 + ['Peptic ulcer diseae'] * 5
}

df = pd.DataFrame(data)

symptom_columns = [col for col in df.columns if col != 'condition']
X = df[symptom_columns]
y = df['condition']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Prescription management
prescriptions = []

# Appointment booking
appointments = []

# Urgency levels
urgency_keywords = {
    'high': ['chest_pain', 'breathlessness', 'high_fever', 'coma', 'acute_liver_failure'],
    'medium': ['vomiting', 'abdominal_pain', 'headache', 'dizziness'],
    'low': ['cough', 'fatigue', 'itching', 'runny_nose']
}

def get_urgency_level(symptoms):
    for symptom in symptoms:
        if symptom in urgency_keywords['high']:
            return 'high'
        elif symptom in urgency_keywords['medium']:
            return 'medium'
    return 'low'

def diagnose_patient(symptoms_dict):
    patient_df = pd.DataFrame([symptoms_dict])
    prediction = model.predict(patient_df)[0]
    probabilities = model.predict_proba(patient_df)[0]
    confidence = max(probabilities) * 100
    
    urgency = get_urgency_level([k for k, v in symptoms_dict.items() if v == 1])
    
    return prediction, confidence, urgency

def user_interface():
    print("Welcome to NHS Diagnostic Assistant")
    print("Please enter your symptoms (type 'done' when finished):")
    
    symptoms = {}
    for col in symptom_columns:
        response = input(f"Do you have {col.replace('_', ' ')}? (y/n): ").lower()
        if response == 'y':
            symptoms[col] = 1
        elif response == 'n':
            symptoms[col] = 0
        elif response == 'done':
            break
    
    if not symptoms:
        print("No symptoms entered.")
        return
    
    # Fill missing symptoms with 0
    for col in symptom_columns:
        if col not in symptoms:
            symptoms[col] = 0
    
    condition, confidence, urgency = diagnose_patient(symptoms)
    
    print(f"\nDiagnosis: {condition}")
    print(f"Confidence: {confidence:.2f}%")
    print(f"Urgency Level: {urgency}")
    
    # Visual output
    plt.figure(figsize=(8, 6))
    symptom_names = [k.replace('_', ' ') for k, v in symptoms.items() if v == 1]
    if symptom_names:
        sns.barplot(x=symptom_names, y=[1]*len(symptom_names))
        plt.title(f"Symptoms for {condition}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    else:
        print("No symptoms to visualize.")
    
    # Appointment booking
    if urgency == 'high':
        print("URGENT: Please book an in-person appointment immediately!")
        appointments.append({'type': 'in-person', 'condition': condition, 'date': datetime.now() + timedelta(days=1)})
    elif urgency == 'medium':
        print("Please book an in-person appointment within 3 days.")
        appointments.append({'type': 'in-person', 'condition': condition, 'date': datetime.now() + timedelta(days=3)})
    else:
        print("For less urgent symptoms, consider a phone consultation.")
        appointments.append({'type': 'phone', 'condition': condition, 'date': datetime.now() + timedelta(days=7)})
    
    print(f"Appointment booked: {appointments[-1]}")

def manage_prescriptions():
    print("\nPrescription Management")
    print("1. View active prescriptions")
    print("2. View expired prescriptions")
    print("3. Add new prescription")
    choice = input("Choose an option: ")
    
    if choice == '1':
        active = [p for p in prescriptions if p['expiry'] > datetime.now()]
        if active:
            for p in active:
                print(f"Medication: {p['medication']}, Expiry: {p['expiry'].strftime('%Y-%m-%d')}")
        else:
            print("No active prescriptions.")
    elif choice == '2':
        expired = [p for p in prescriptions if p['expiry'] <= datetime.now()]
        if expired:
            for p in expired:
                print(f"Medication: {p['medication']}, Expiry: {p['expiry'].strftime('%Y-%m-%d')}")
        else:
            print("No expired prescriptions.")
    elif choice == '3':
        med = input("Medication name: ")
        expiry_str = input("Expiry date (YYYY-MM-DD): ")
        expiry = datetime.strptime(expiry_str, '%Y-%m-%d')
        prescriptions.append({'medication': med, 'expiry': expiry})
        print("Prescription added.")

# Main menu
while True:
    print("\nMain Menu:")
    print("1. Diagnose symptoms")
    print("2. Manage prescriptions")
    print("3. View appointments")
    print("4. Exit")
    choice = input("Choose an option: ")
    
    if choice == '1':
        user_interface()
    elif choice == '2':
        manage_prescriptions()
    elif choice == '3':
        if appointments:
            for appt in appointments:
                print(f"Type: {appt['type']}, Condition: {appt['condition']}, Date: {appt['date'].strftime('%Y-%m-%d')}")
        else:
            print("No appointments booked.")
    elif choice == '4':
        break
    else:
        print("Invalid choice.")

# Test with sample
sample_symptoms = {'itching': 1, 'skin_rash': 1, 'nodal_skin_eruptions': 1, 'continuous_sneezing': 0, 'shivering': 0}
condition, confidence, urgency = diagnose_patient(sample_symptoms)
print(f"\nSample Diagnosis: {condition}, Confidence: {confidence:.2f}%, Urgency: {urgency}")