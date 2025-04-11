import pandas as pd
import numpy as np
from scipy import stats
import os
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import chi2_contingency, f_oneway, ttest_ind

def load_data(data_dir='data'):
    baseline_df = pd.read_csv(f'{data_dir}/baseline_survey.csv')
    treatment_df = pd.read_csv(f'{data_dir}/treatment_assignment.csv')
    endline_df = pd.read_csv(f'{data_dir}/endline_survey.csv')
    return baseline_df, treatment_df, endline_df

def merge_datasets(baseline_df, treatment_df, endline_df):
    merged_df = pd.merge(baseline_df, treatment_df, on='participant_id')
    merged_df['completed_endline'] = merged_df['participant_id'].isin(endline_df['participant_id']).astype(int)
    merged_df = pd.merge(merged_df, endline_df, on='participant_id', how='left')

    # set missing vaccine uptake to zero to stop NaN warnings
    merged_df['vaccine_uptake'] = merged_df['vaccine_uptake'].fillna(0)

    return merged_df

def generate_descriptive_stats(merged_df):
    demo_stats = {
        'Variable': [],
        'Category': [],
        'Count': [],
        'Percentage/Mean': [],
        'SD': []
    }

    # Categorical variables
    for var in ['gender', 'region', 'education', 'income', 'political_leaning']:
        value_counts = merged_df[var].value_counts()
        for category, count in value_counts.items():
            demo_stats['Variable'].append(var)
            demo_stats['Category'].append(category)
            demo_stats['Count'].append(count)
            demo_stats['Percentage/Mean'].append(count / len(merged_df) * 100)
            demo_stats['SD'].append(None)

    # Continuous variables
    for var in ['age', 'baseline_vaccine_confidence']:
        demo_stats['Variable'].append(var)
        demo_stats['Category'].append(None)
        demo_stats['Count'].append(merged_df[var].count())
        demo_stats['Percentage/Mean'].append(merged_df[var].mean())
        demo_stats['SD'].append(merged_df[var].std())

    return pd.DataFrame(demo_stats)

def check_randomization(merged_df):
    randomization_checks = {'Variable': [], 'Test': [], 'Statistic': [], 'p_value': []}

    categorical_vars = ['gender', 'region', 'education', 'income', 'political_leaning']
    for var in categorical_vars:
        contingency_table = pd.crosstab(merged_df[var], merged_df['treatment_group'])
        chi2, p_val, _, _ = chi2_contingency(contingency_table)
        randomization_checks['Variable'].append(var)
        randomization_checks['Test'].append('Chi-square')
        randomization_checks['Statistic'].append(chi2)
        randomization_checks['p_value'].append(p_val)

    continuous_vars = ['age', 'baseline_vaccine_confidence']
    for var in continuous_vars:
        groups = [merged_df[merged_df['treatment_group'] == group][var] for group in merged_df['treatment_group'].unique()]
        f_stat, p_val = f_oneway(*groups)
        randomization_checks['Variable'].append(var)
        randomization_checks['Test'].append('ANOVA')
        randomization_checks['Statistic'].append(f_stat)
        randomization_checks['p_value'].append(p_val)

    return pd.DataFrame(randomization_checks)

def analyze_attrition(merged_df):
    attrition_results = {'Variable': [], 'Test': [], 'Statistic': [], 'p_value': []}

    categorical_vars = ['gender', 'region', 'education', 'income', 'political_leaning', 'treatment_group']
    for var in categorical_vars:
        contingency_table = pd.crosstab(merged_df[var], merged_df['completed_endline'])
        chi2, p_val, _, _ = chi2_contingency(contingency_table)
        attrition_results['Variable'].append(var)
        attrition_results['Test'].append('Chi-square')
        attrition_results['Statistic'].append(chi2)
        attrition_results['p_value'].append(p_val)

    continuous_vars = ['age', 'baseline_vaccine_confidence']
    for var in continuous_vars:
        completed = merged_df[merged_df['completed_endline'] == 1][var]
        dropped = merged_df[merged_df['completed_endline'] == 0][var]
        t_stat, p_val = ttest_ind(completed, dropped, nan_policy='omit')
        attrition_results['Variable'].append(var)
        attrition_results['Test'].append('t-test')
        attrition_results['Statistic'].append(t_stat)
        attrition_results['p_value'].append(p_val)

    return pd.DataFrame(attrition_results)

def analyze_treatment_effects(merged_df):
    analysis_df = merged_df[merged_df['completed_endline'] == 1].copy()

    # Simple base model
    try:
        formula = 'vaccine_uptake ~ C(treatment_group, Treatment("control"))'
        base_model = smf.logit(formula, data=analysis_df).fit(disp=0)
    except:
        # Fall back to OLS if logit fails to converge
        try:
            formula = 'vaccine_uptake ~ C(treatment_group, Treatment("control"))'
            base_model = smf.ols(formula, data=analysis_df).fit()
        except:
            base_model = "Base model failed to converge"

    # Model with demographic controls
    try:
        formula = ('vaccine_uptake ~ C(treatment_group, Treatment("control")) + '
                   'age + C(gender) + C(education) + C(region) + '
                   'C(political_leaning) + baseline_vaccine_confidence')
        adjusted_model = smf.logit(formula, data=analysis_df).fit(disp=0)
    except:
        try:
            adjusted_model = smf.ols(formula, data=analysis_df).fit()
        except:
            adjusted_model = "Adjusted model failed to converge"

    # Generate summary statistics for visualization
    treatment_stats = analysis_df.groupby('treatment_group')['vaccine_uptake'].agg(['mean', 'count'])
    treatment_stats['se'] = np.sqrt(treatment_stats['mean'] * (1 - treatment_stats['mean']) / treatment_stats['count'])
    treatment_stats['ci_lower'] = treatment_stats['mean'] - 1.96 * treatment_stats['se']
    treatment_stats['ci_upper'] = treatment_stats['mean'] + 1.96 * treatment_stats['se']
    treatment_stats['percentage'] = treatment_stats['mean'] * 100

    return {
        'base_model': base_model,
        'adjusted_model': adjusted_model,
        'treatment_stats': treatment_stats
    }

def analyze_heterogeneous_effects(merged_df):
    analysis_df = merged_df[merged_df['completed_endline'] == 1].copy()

    # Education-stratified statistics for visualization
    edu_stats = analysis_df.groupby(['education', 'treatment_group'])['vaccine_uptake'].agg(['mean', 'count'])
    edu_stats['se'] = np.sqrt(edu_stats['mean'] * (1 - edu_stats['mean']) / edu_stats['count'])
    edu_stats['ci_lower'] = edu_stats['mean'] - 1.96 * edu_stats['se']
    edu_stats['ci_upper'] = edu_stats['mean'] + 1.96 * edu_stats['se']
    edu_stats['percentage'] = edu_stats['mean'] * 100

    # Interaction model
    try:
        formula = 'vaccine_uptake ~ C(treatment_group) * C(education)'
        interaction_model = smf.logit(formula, data=analysis_df).fit(disp=0)
    except:
        try:
            interaction_model = smf.ols(formula, data=analysis_df).fit()
        except:
            interaction_model = "Interaction model failed to converge"

    # Stratified by education level - simplified models
    education_models = {}
    for edu in ['High School', 'Some College', 'Bachelor', 'Graduate']:
        if edu in analysis_df['education'].unique():
            edu_df = analysis_df[analysis_df['education'] == edu].copy()
            if len(edu_df) >= 30:
                try:
                    edu_model = smf.logit('vaccine_uptake ~ C(treatment_group)', data=edu_df).fit(disp=0)
                    education_models[edu] = edu_model
                except:
                    try:
                        edu_model = smf.ols('vaccine_uptake ~ C(treatment_group)', data=edu_df).fit()
                        education_models[edu] = edu_model
                    except:
                        education_models[edu] = "Model failed to converge"
            else:
                education_models[edu] = "Insufficient data points"
        else:
            education_models[edu] = "Education level not in dataset"

    return {
        'interaction_model': interaction_model,
        'education_models': education_models,
        'education_stats': edu_stats
    }

def analyze_confidence_change(merged_df):
    analysis_df = merged_df[merged_df['completed_endline'] == 1].copy()
    analysis_df['confidence_change'] = analysis_df['endline_vaccine_confidence'] - analysis_df['baseline_vaccine_confidence']

    # ANCOVA model
    try:
        formula = 'endline_vaccine_confidence ~ C(treatment_group) + baseline_vaccine_confidence'
        ancova_model = smf.ols(formula, data=analysis_df).fit()
    except:
        ancova_model = "ANCOVA model failed"

    # t-tests for each group
    within_group_tests = {}
    for group in analysis_df['treatment_group'].unique():
        group_df = analysis_df[analysis_df['treatment_group'] == group].copy()
        t_stat, p_val = stats.ttest_rel(group_df['endline_vaccine_confidence'],
                                       group_df['baseline_vaccine_confidence'])
        within_group_tests[group] = {'t_statistic': t_stat, 'p_value': p_val}

    # Summary statistics for visualization
    confidence_stats = analysis_df.groupby('treatment_group').agg(
        baseline_mean=('baseline_vaccine_confidence', 'mean'),
        baseline_std=('baseline_vaccine_confidence', 'std'),
        baseline_count=('baseline_vaccine_confidence', 'count'),
        endline_mean=('endline_vaccine_confidence', 'mean'),
        endline_std=('endline_vaccine_confidence', 'std'),
        endline_count=('endline_vaccine_confidence', 'count')
    )

    confidence_stats['baseline_se'] = confidence_stats['baseline_std'] / np.sqrt(confidence_stats['baseline_count'])
    confidence_stats['endline_se'] = confidence_stats['endline_std'] / np.sqrt(confidence_stats['endline_count'])
    confidence_stats['baseline_ci_lower'] = confidence_stats['baseline_mean'] - 1.96 * confidence_stats['baseline_se']
    confidence_stats['baseline_ci_upper'] = confidence_stats['baseline_mean'] + 1.96 * confidence_stats['baseline_se']
    confidence_stats['endline_ci_lower'] = confidence_stats['endline_mean'] - 1.96 * confidence_stats['endline_se']
    confidence_stats['endline_ci_upper'] = confidence_stats['endline_mean'] + 1.96 * confidence_stats['endline_se']

    return {
        'ancova_model': ancova_model,
        'within_group_tests': within_group_tests,
        'confidence_stats': confidence_stats
    }

def generate_political_confidence_stats(merged_df):
    political_order = ["Very Liberal", "Liberal", "Moderate", "Conservative", "Very Conservative"]

    # Ensure we use the correct order for political leanings
    political_stats = merged_df.groupby('political_leaning')['baseline_vaccine_confidence'].agg(['mean', 'std', 'count'])
    political_stats = political_stats.reindex(political_order)

    political_stats['se'] = political_stats['std'] / np.sqrt(political_stats['count'])
    political_stats['ci_lower'] = political_stats['mean'] - 1.96 * political_stats['se']
    political_stats['ci_upper'] = political_stats['mean'] + 1.96 * political_stats['se']

    return political_stats

def generate_key_findings_stats(merged_df, treatment_effects, heterogeneous_effects):
    analysis_df = merged_df[merged_df['completed_endline'] == 1].copy()

    key_findings = {
        'Comparison': [],
        'Effect': [],
        'CI_Lower': [],
        'CI_Upper': []
    }

    # Reason vs Control
    reason_effect = treatment_effects['treatment_stats'].loc['reason', 'percentage'] - \
                   treatment_effects['treatment_stats'].loc['control', 'percentage']
    reason_se = np.sqrt(
        treatment_effects['treatment_stats'].loc['reason', 'se']**2 +
        treatment_effects['treatment_stats'].loc['control', 'se']**2
    ) * 100

    key_findings['Comparison'].append('Reason vs. Control')
    key_findings['Effect'].append(reason_effect)
    key_findings['CI_Lower'].append(reason_effect - 1.96 * reason_se)
    key_findings['CI_Upper'].append(reason_effect + 1.96 * reason_se)

    # Emotion vs Control
    emotion_effect = treatment_effects['treatment_stats'].loc['emotion', 'percentage'] - \
                    treatment_effects['treatment_stats'].loc['control', 'percentage']
    emotion_se = np.sqrt(
        treatment_effects['treatment_stats'].loc['emotion', 'se']**2 +
        treatment_effects['treatment_stats'].loc['control', 'se']**2
    ) * 100

    key_findings['Comparison'].append('Emotion vs. Control')
    key_findings['Effect'].append(emotion_effect)
    key_findings['CI_Lower'].append(emotion_effect - 1.96 * emotion_se)
    key_findings['CI_Upper'].append(emotion_effect + 1.96 * emotion_se)

    # Education-specific effects
    edu_levels = ['High School', 'Graduate']

    for edu in edu_levels:
        if edu in heterogeneous_effects['education_stats'].index.get_level_values('education'):
            if (edu, 'reason') in heterogeneous_effects['education_stats'].index and \
               (edu, 'control') in heterogeneous_effects['education_stats'].index:

                edu_reason = heterogeneous_effects['education_stats'].loc[(edu, 'reason'), 'percentage'] - \
                            heterogeneous_effects['education_stats'].loc[(edu, 'control'), 'percentage']
                edu_se = np.sqrt(
                    heterogeneous_effects['education_stats'].loc[(edu, 'reason'), 'se']**2 +
                    heterogeneous_effects['education_stats'].loc[(edu, 'control'), 'se']**2
                ) * 100

                key_findings['Comparison'].append(f'Reason Effect: {edu}')
                key_findings['Effect'].append(edu_reason)
                key_findings['CI_Lower'].append(edu_reason - 1.96 * edu_se)
                key_findings['CI_Upper'].append(edu_reason + 1.96 * edu_se)

    return pd.DataFrame(key_findings)

def run_analysis(data_dir='data'):
    baseline_df, treatment_df, endline_df = load_data(data_dir)
    merged_df = merge_datasets(baseline_df, treatment_df, endline_df)

    # Generate all statistics and models
    descriptive_stats = generate_descriptive_stats(merged_df)
    randomization_check = check_randomization(merged_df)
    attrition_analysis = analyze_attrition(merged_df)
    treatment_effects = analyze_treatment_effects(merged_df)
    heterogeneous_effects = analyze_heterogeneous_effects(merged_df)
    confidence_analysis = analyze_confidence_change(merged_df)

    # Generate visualization-specific statistics
    political_confidence = generate_political_confidence_stats(merged_df)
    key_findings = generate_key_findings_stats(merged_df, treatment_effects, heterogeneous_effects)

    return {
        'descriptive_stats': descriptive_stats,
        'randomization_check': randomization_check,
        'attrition_analysis': attrition_analysis,
        'treatment_effects': treatment_effects,
        'heterogeneous_effects': heterogeneous_effects,
        'confidence_analysis': confidence_analysis,
        'political_confidence': political_confidence,
        'key_findings': key_findings
    }

if __name__ == "__main__":
    results = run_analysis()
    print("Analysis completed successfully.")
