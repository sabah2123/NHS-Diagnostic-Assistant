# NHS-Diagnostic-Assistant

An AI-powered diagnostic tool that analyses patient symptoms and predicts potential conditions, helping reduce NHS triage times and improve patient referral accuracy using machine learning.

## Features

- **Expanded Dataset**: Uses a larger dataset with multiple symptoms and conditions, simulating real Kaggle datasets.
- **User Input Interface**: Interactive command-line interface where users can type their symptoms.
- **Confidence Scores**: Displays how certain the model is in its diagnosis.
- **Visual Output**: Simple bar chart visualization of reported symptoms.
- **Appointment Booking**: Automatically books appointments based on urgency levels (high urgency for in-person, low for phone consultations).
- **Prescription Management**: View and manage active and expired prescriptions.

- ## Business Impact
- This tool could reduce NHS triage times, free up GP appointments, and improve patient outcomes.

## Installation

1. Clone the repository.
2. Install dependencies: `pip install pandas scikit-learn matplotlib seaborn`
3. Run the application: `python main.py`

## Usage

Run the script and follow the menu options:
1. Diagnose symptoms: Answer yes/no to symptom questions.
2. Manage prescriptions: Add or view prescriptions.
3. View appointments: See booked appointments.
4. Exit.

## Dataset

The application uses a synthetic dataset with 10 symptoms and 6 conditions for demonstration. In a real-world scenario, this would be replaced with actual medical data from sources like Kaggle's "Disease Symptom Prediction" dataset.

## Model

Uses Decision Tree Classifier from scikit-learn for prediction.
