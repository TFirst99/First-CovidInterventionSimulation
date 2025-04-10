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
    vaccine_intention = [generate_vaccine_confidence(e, p) for e, p in zip(education, political_leaning)]

    baseline_df = pd.DataFrame({
        'participant_id': participant_ids,
        'age': age,
        'gender': gender,
        'region': region,
        'education': education,
        'income': income,
        'baseline_vaccine_intention': vaccine_intention
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

    # randomly select participants to complete the endline survey
    n_endline = int(n_participants * response_rate)
    endline_participants = np.random.choice(merged_df['participant_id'], n_endline, replace=False)

    endline_df = pd.DataFrame({'participant_id': endline_participants})

    endline_with_treatment = pd.merge(endline_df, treatment_df, on='participant_id')

    # define the uptake of the vaccine based on treatment
    base_prob = 0.5
    reason_effect = 0.1
    emotion_effect = 0.15

    # apply the effect of treatment
    probs = np.where(endline_with_treatment['treatment_group'] == 'control', base_prob,
                    np.where(endline_with_treatment['treatment_group'] == 'reason',
                             base_prob + reason_effect, base_prob + emotion_effect))

    vaccine_uptake = np.random.binomial(1, probs)

    endline_df['vaccine_uptake'] = vaccine_uptake

    # follow-up vaccine intention
    follow_up_intention = np.random.normal(4, 1.5, n_endline)
    follow_up_intention = np.clip(follow_up_intention, 1, 7).round(1)

    # make it higher for those who got vaccinated
    follow_up_intention = np.where(vaccine_uptake == 1,
                                  follow_up_intention + 1.5,
                                  follow_up_intention)
    follow_up_intention = np.clip(follow_up_intention, 1, 7).round(1)

    endline_df['follow_up_intention'] = follow_up_intention

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
