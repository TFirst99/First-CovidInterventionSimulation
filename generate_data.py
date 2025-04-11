import pandas as pd
import numpy as np
import os

def generate_political_leaning(gender, age, region):
    # Roughly based on national averages
    base_probs = {
        "Very Liberal": 0.1,
        "Liberal": 0.18,
        "Moderate": 0.35,
        "Conservative": 0.27,
        "Very Conservative": 0.1
    }

    score = 0

    if gender == "Female":
        score += 0.05

    score -= (age - 40) * 0.005

    if region == "Northeast":
        score += 0.10
    elif region == "West":
        score += 0.08
    elif region == "Midwest":
        score -= 0.03
    elif region == "South":
        score -= 0.12

    adjusted_probs = {
        "Very Liberal": max(0.001, base_probs["Very Liberal"] + score * 0.3),
        "Liberal": max(0.001, base_probs["Liberal"] + score * 0.2),
        "Moderate": max(0.001, base_probs["Moderate"] - score * 0.1),
        "Conservative": max(0.001, base_probs["Conservative"] - score * 0.2),
        "Very Conservative": max(0.001, base_probs["Very Conservative"] - score * 0.2)
    }

    total = sum(adjusted_probs.values())
    normalized_probs = [adjusted_probs[leaning]/total for leaning in ["Very Liberal", "Liberal", "Moderate", "Conservative", "Very Conservative"]]

    return np.random.choice(
        ["Very Liberal", "Liberal", "Moderate", "Conservative", "Very Conservative"],
        p=normalized_probs
    )

def generate_education(gender, age, political_leaning):
    # Base probabilities (national averages)
    base_probs = {
        "High School": 0.28,
        "Some College": 0.25,
        "Bachelor": 0.23,
        "Graduate": 0.14
    }
    score = 0
    # Gender modifier
    if gender == "Female":
        score += 0.05
    # Age modifier
    score -= (age - 40) * 0.004

    if political_leaning in ["Very Liberal", "Liberal"]:
        score += 0.10
    elif political_leaning in ["Conservative", "Very Conservative"]:
        score -= 0.08

    adjusted_probs = {
        "High School": base_probs["High School"] - score * 0.4,
        "Some College": base_probs["Some College"] - score * 0.1,
        "Bachelor": base_probs["Bachelor"] + score * 0.2,
        "Graduate": base_probs["Graduate"] + score * 0.3
    }

    # normalize probabilities
    total = sum(adjusted_probs.values())
    normalized_probs = [adjusted_probs[level]/total for level in ["High School", "Some College", "Bachelor", "Graduate"]]

    return np.random.choice(
        ["High School", "Some College", "Bachelor", "Graduate"],
        p=normalized_probs
    )

def generate_income(education, age):
    if education == "Graduate":
        base_probabilities = [0.05, 0.1, 0.3, 0.35, 0.2]
    elif education == "Bachelor":
        base_probabilities = [0.1, 0.2, 0.3, 0.25, 0.15]
    elif education == "Some College":
        base_probabilities = [0.15, 0.25, 0.3, 0.2, 0.1]
    elif education == "High School":
        base_probabilities = [0.25, 0.3, 0.25, 0.15, 0.05]

    age_factor = max(0.1, 1 - 0.0005 * (age - 50)**2)
    # Adjust the base probabilities based on age using a quadratic function
    adjusted_probs = []
    for i, p in enumerate(base_probabilities):
        if i < 2:
            adj = p * (2 - age_factor)
        else:
            adj = p * age_factor * 1.5
        adjusted_probs.append(adj)

    adjusted_probs = [p/sum(adjusted_probs) for p in adjusted_probs]

    return np.random.choice(
        ["Under $30k", "$30k-$60k", "$60k-$100k", "$100k-$150k", "Over $150k"],
        p=adjusted_probs
    )

def generate_vaccine_confidence(education, political_leaning):
    base_value = np.random.normal(4, 1)

    education_modifier = {
        "High School": -0.5,
        "Some College": 0,
        "Bachelor": 0.5,
        "Graduate": 0.5
    }

    political_modifier = {
        "Very Liberal": 0.7,
        "Liberal": 0.5,
        "Moderate": 0,
        "Conservative": -0.5,
        "Very Conservative": -0.7
    }

    value = base_value + education_modifier[education] + political_modifier[political_leaning]

    return max(1, min(7, round(value, 1)))

def generate_baseline_data(n_participants=5000, seed=42):
    np.random.seed(seed)

    # Create unique identifiers
    participant_ids = np.arange(1, n_participants + 1)

    # Create background demographic data for each participant
    age = np.random.normal(45, 15, n_participants).astype(int)
    age = np.clip(age, 18, 100)
    gender = np.random.choice(['Male', 'Female'], n_participants, p=[0.495, 0.505])
    region = np.random.choice(['Northeast', 'Midwest', 'South', 'West'], n_participants)

    # Generate characteristics based on background and other characteristics
    political_leaning = [generate_political_leaning(g, a, r) for g, a, r in zip(gender, age, region)]
    education = [generate_education(g, a, p) for g, a, p in zip(gender, age, political_leaning)]
    income = [generate_income(e, a) for e, a in zip(education, age)]
    baseline_vaccine_confidence = [generate_vaccine_confidence(e, p) for e, p in zip(education, political_leaning)]

    baseline_df = pd.DataFrame({
        'participant_id': participant_ids,
        'age': age,
        'gender': gender,
        'region': region,
        'education': education,
        'income': income,
        'baseline_vaccine_confidence': baseline_vaccine_confidence
    })

    return baseline_df

def generate_treatment_assignment(baseline_df, seed=42):
    np.random.seed(seed)

    n_participants = len(baseline_df)
    participant_ids = baseline_df['participant_id'].values

    treatments = np.random.choice(['reason', 'emotion', 'control'], n_participants, p=[1/3, 1/3, 1/3])

    treatment_df = pd.DataFrame({
        'participant_id': participant_ids,
        'treatment_group': treatments
    })

    return treatment_df

def generate_endline_data(baseline_df, treatment_df, response_rate=0.9, seed=42):
    np.random.seed(seed)

    merged_df = pd.merge(baseline_df, treatment_df, on='participant_id')
    n_participants = len(merged_df)

    # Only 4500 participants complete the endline survey
    n_endline = int(n_participants * response_rate)
    endline_participants = np.random.choice(merged_df['participant_id'], n_endline, replace=False)

    endline_df = pd.DataFrame({'participant_id': endline_participants})

    endline_with_data = pd.merge(
        endline_df,
        merged_df[['participant_id', 'baseline_vaccine_confidence', 'treatment_group',
                  'education', 'age']],
        on='participant_id'
    )

    # ---- MODEL VACCINATION UPTAKE ----
    # Baseline uptake converted to [0,1]
    base_prob = (endline_with_data['baseline_vaccine_confidence'] - 1) / 6

    # Age effect
    age_factor = (endline_with_data['age'] - 18) / (80 - 18) * 0.2

    baseline_uptake = base_prob + age_factor
    baseline_uptake = np.clip(baseline_uptake, 0.1, 0.9)

    # Treatment effects
    treatment_effect = np.zeros(len(endline_with_data))
    reason_effect = 0.10
    emotion_effect = 0.12

    # Education moderates treatment effect
    edu_modifier = np.ones(len(endline_with_data))
    edu_modifier[endline_with_data['education'] == 'High School'] = 0.7
    edu_modifier[endline_with_data['education'] == 'Some College'] = 0.9
    edu_modifier[endline_with_data['education'] == 'Bachelor'] = 1.2
    edu_modifier[endline_with_data['education'] == 'Graduate'] = 1.5

    is_reason = endline_with_data['treatment_group'] == 'reason'
    is_emotion = endline_with_data['treatment_group'] == 'emotion'

    # Reason is more affected by education level
    treatment_effect[is_reason] = reason_effect * edu_modifier[is_reason]

    # Emotion is less affected by education level
    emotion_edu_modifier = (edu_modifier + 2) / 3
    treatment_effect[is_emotion] = emotion_effect * emotion_edu_modifier[is_emotion]

    uptake_prob = baseline_uptake + treatment_effect
    uptake_prob += np.random.normal(0, 0.05, len(endline_with_data))
    uptake_prob = np.clip(uptake_prob, 0.01, 0.99)

    vaccine_uptake = np.random.binomial(1, uptake_prob)
    endline_with_data['vaccine_uptake'] = vaccine_uptake

    # ***** Endline Confidence *****

    endline_confidence = endline_with_data['baseline_vaccine_confidence'].values.copy()
    got_treatment = (is_reason | is_emotion)
    control_group = ~got_treatment

    # Case 1: Increase confidence if they got vaccinated
    vaccine_boost = np.random.uniform(1.0, 2.0, len(endline_with_data))
    endline_confidence[vaccine_uptake == 1] += vaccine_boost[vaccine_uptake == 1]

    # Case 2: Decrease if they saw an ad, but didn't get vaccinated
    ad_no_vax_penalty = np.random.uniform(0.5, 1.5, len(endline_with_data))
    ad_no_vax = (got_treatment) & (vaccine_uptake == 0)
    endline_confidence[ad_no_vax] -= ad_no_vax_penalty[ad_no_vax]

    # Case 3: Control group who didn't get vaccinated stays roughly the same
    control_no_vax = (control_group) & (vaccine_uptake == 0)
    control_change = np.random.uniform(-0.3, 0.3, len(endline_with_data))
    endline_confidence[control_no_vax] += control_change[control_no_vax]

    endline_confidence += np.random.normal(0, 0.2, len(endline_with_data))
    endline_confidence = np.clip(endline_confidence, 1, 7).round(1)

    endline_with_data['endline_vaccine_confidence'] = endline_confidence
    endline_df = endline_with_data[['participant_id', 'vaccine_uptake', 'endline_vaccine_confidence']]

    return endline_df

def save_datasets(baseline_df, treatment_df, endline_df, data_dir='data'):
    os.makedirs(data_dir, exist_ok=True)

    baseline_df.to_csv(f'{data_dir}/baseline_survey.csv', index=False)
    treatment_df.to_csv(f'{data_dir}/treatment_assignment.csv', index=False)
    endline_df.to_csv(f'{data_dir}/endline_survey.csv', index=False)

    print(f"Datasets saved to {data_dir}/")

if __name__ == "__main__":
    baseline_df = generate_baseline_data()
    treatment_df = generate_treatment_assignment(baseline_df)
    endline_df = generate_endline_data(baseline_df, treatment_df)
    save_datasets(baseline_df, treatment_df, endline_df)
