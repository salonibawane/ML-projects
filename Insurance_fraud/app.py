from flask import Flask, render_template, request
import pickle
import numpy as np
from utils import preprocess_data

app = Flask(__name__)

# Load the pre-trained model
with open('insurance_fraud.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Define merged features with their options/ranges
merged_features = {
    'months_as_customer': {'label': 'Months as Customer', 'type': 'range', 'min': 0, 'max': 600},
    'policy_deductable': {'label': 'Policy Deductible', 'type': 'options', 'options': [500, 1000, 2000]},
    'umbrella_limit': {'label': 'Umbrella Limit', 'type': 'options', 'options': [0, 1000000, 2000000, 3000000]},
    # Add more merged features as needed
}

# All feature names including merged features
feature_names = list(merged_features.keys())

# All feature names including merged features and additional non-merged features
all_feature_names = [
    'months_as_customer', 'policy_deductable', 'umbrella_limit',
    'capital-gains', 'capital-loss', 'incident_hour_of_the_day',
    'number_of_vehicles_involved', 'bodily_injuries', 'witnesses',
    'injury_claim', 'property_claim', 'vehicle_claim', 'age', 'insured_sex',
    'property_damage', 'police_report_available', 'total_claim_amount',
    'insured_education_level_College', 'insured_education_level_High School',
    'insured_education_level_JD', 'insured_education_level_MD',
    'insured_education_level_Masters', 'insured_education_level_PhD',
    'insured_occupation_armed-forces', 'insured_occupation_craft-repair',
    'insured_occupation_exec-managerial', 'insured_occupation_farming-fishing',
    'insured_occupation_handlers-cleaners', 'insured_occupation_machine-op-inspct',
    'insured_occupation_other-service', 'insured_occupation_priv-house-serv',
    'insured_occupation_prof-specialty', 'insured_occupation_protective-serv',
    'insured_occupation_sales', 'insured_occupation_tech-support',
    'insured_occupation_transport-moving', 'insured_relationship_not-in-family',
    'insured_relationship_other-relative', 'insured_relationship_own-child',
    'insured_relationship_unmarried', 'insured_relationship_wife',
    'incident_type_Parked Car', 'incident_type_Single Vehicle Collision',
    'incident_type_Vehicle Theft', 'collision_type_Rear Collision',
    'collision_type_Side Collision', 'incident_severity_Minor Damage',
    'incident_severity_Total Loss', 'incident_severity_Trivial Damage',
    'authorities_contacted_Fire', 'authorities_contacted_Other',
    'authorities_contacted_Police'
]

@app.route('/')
def home():
    return render_template('index.html', merged_features=merged_features)


@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from the form
    input_features = []
    for feature, params in merged_features.items():
        if params['type'] == 'range':
            input_features.append(float(request.form.get(feature)))
        elif params['type'] == 'options':
            option = float(request.form.get(feature))
            input_features.extend([1 if option == opt else 0 for opt in params['options']])

    # Add zeros for non-merged features
    for feature in all_feature_names:
        if feature not in merged_features:
            input_features.append(0.0)

    # Convert input values to float and handle missing values
    input_features = [0.0 if np.isnan(value) else value for value in input_features]

    # Only keep the first 52 features (the ones expected by the model)
    input_features = input_features[:52]

    # Preprocess the input
    features_array = np.array(input_features).reshape(1, -1)
    processed_features = preprocess_data(features_array)

    # Make prediction probabilities
    prediction_prob = model.predict_proba(processed_features)[0]

    # Set threshold for deciding "NO" result
    threshold = 0.5  # You can adjust this threshold based on your preference

    # Convert prediction probabilities to "YES" or "NO"
    if prediction_prob[1] > threshold:
        prediction_label = "YES"
    else:
        prediction_label = "NO"

    # Additional information for display
    additional_info = {
        'Months as Customer': input_features[0],
        'Policy Deductible': input_features[1],
        'Umbrella Limit': input_features[2],
        'Bodily Injuries': input_features[7],
        'Witnesses': input_features[8],
        'Police Report Available': input_features[15],
        'Total Claim Amount': input_features[16],
        'Injury Claim': input_features[9],
        'Property Claim': input_features[10],
        'Vehicle Claim': input_features[11],
        'Age': input_features[12]
    }

    # Render result template with prediction and additional information
    return render_template('result.html', prediction=prediction_label, additional_info=additional_info)


if __name__ == '__main__':
    app.run(debug=True)


