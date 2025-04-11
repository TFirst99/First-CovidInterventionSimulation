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
    merged_df = pd.merge(merged_df, endline_df, on='participant_id', how='left')

    # Replace NaN values in vaccine_uptake with 0
    merged_df['vaccine_uptake'] = merged_df['vaccine_uptake'].fillna(0)

    return merged_df

def analyze_effects(merged_df):
    vax_by_treatment = merged_df.groupby('treatment_group')['vaccine_uptake'].mean()

    # statistical tests
    reason_group = merged_df[merged_df['treatment_group'] == 'reason']['vaccine_uptake']
    emotion_group = merged_df[merged_df['treatment_group'] == 'emotion']['vaccine_uptake']
    control_group = merged_df[merged_df['treatment_group'] == 'control']['vaccine_uptake']

    reason_vs_control = stats.ttest_ind(reason_group, control_group)
    emotion_vs_control = stats.ttest_ind(emotion_group, control_group)

    # CHECK THIS TEST
    def cohens_d(group1, group2):
        n1, n2 = len(group1), len(group2)
        mean1, mean2 = np.mean(group1), np.mean(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        pooled_sd = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        return (mean1 - mean2) / pooled_sd

    reason_effect_size = cohens_d(reason_group, control_group)
    emotion_effect_size = cohens_d(emotion_group, control_group)

    results = {
        'vaccine_rates': vax_by_treatment,
        'reason_vs_control': {
            't_stat': reason_vs_control.statistic,
            'p_value': reason_vs_control.pvalue,
            'effect_size': reason_effect_size
        },
        'emotion_vs_control': {
            't_stat': emotion_vs_control.statistic,
            'p_value': emotion_vs_control.pvalue,
            'effect_size': emotion_effect_size
        }
    }

    return results

def save_results(merged_df, results, data_dir='data', results_dir='results/tables'):
    os.makedirs(results_dir, exist_ok=True)

    merged_df.to_csv(f'{data_dir}/merged_data.csv', index=False)

    summary_by_group = merged_df.groupby('treatment_group').agg({
        'vaccine_uptake': ['mean', 'std', 'count'],
        'follow_up_intention': ['mean', 'std']
    })

    summary_by_group.to_csv(f'{results_dir}/summary_by_treatment.csv')

    test_results = pd.DataFrame({
        'Comparison': ['Reason vs Control', 'Emotion vs Control'],
        't_statistic': [results['reason_vs_control']['t_stat'],
                       results['emotion_vs_control']['t_stat']],
        'p_value': [results['reason_vs_control']['p_value'],
                   results['emotion_vs_control']['p_value']],
        'effect_size': [results['reason_vs_control']['effect_size'],
                       results['emotion_vs_control']['effect_size']]
    })

    test_results.to_csv(f'{results_dir}/statistical_tests.csv', index=False)

    print(f"Analysis results saved to {results_dir}/")

    return summary_by_group, test_results

if __name__ == "__main__":
    baseline_df, treatment_df, endline_df = load_data()
    merged_df = merge_datasets(baseline_df, treatment_df, endline_df)
    results = analyze_effects(merged_df)
    summary_by_group, test_results = save_results(merged_df, results)

    print("\nvaccine uptake by treatment group:")
    print(results['vaccine_rates'])

    print("\nstatistical test results:")
    print(test_results)
