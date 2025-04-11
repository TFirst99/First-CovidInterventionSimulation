import pandas as pd
import numpy as np
import os
from analyze_data import run_analysis, load_data, merge_datasets

def setup_table_folder():
    os.makedirs('tables', exist_ok=True)

def format_p_value(p_value):
    if p_value < 0.001:
        return "<0.001"
    else:
        return f"{p_value:.3f}"

def create_demographics_table(descriptive_stats):
    demo_table = pd.DataFrame(columns=['Variable', 'Category', 'Count', 'Percentage/Mean', 'SD'])

    for var_name in descriptive_stats['Variable'].unique():
        var_rows = descriptive_stats[descriptive_stats['Variable'] == var_name]

        if var_rows['Category'].isna().all():
            row = var_rows.iloc[0]
            new_row = pd.DataFrame([{
                'Variable': row['Variable'],
                'Category': '',
                'Count': f"{row['Count']:.0f}",
                'Percentage/Mean': f"{row['Percentage/Mean']:.2f}",
                'SD': f"{row['SD']:.2f}"
            }])
            demo_table = pd.concat([demo_table, new_row], ignore_index=True)
        else:
            for _, row in var_rows.iterrows():
                new_row = pd.DataFrame([{
                    'Variable': row['Variable'],
                    'Category': row['Category'],
                    'Count': f"{row['Count']:.0f}",
                    'Percentage/Mean': f"{row['Percentage/Mean']:.1f}%",
                    'SD': ''
                }])
                demo_table = pd.concat([demo_table, new_row], ignore_index=True)

    demo_table.to_csv('tables/demographics_table.csv', index=False)

    return demo_table

def create_balance_table(merged_df):
    treatment_groups = merged_df['treatment_group'].unique()

    balance_table = pd.DataFrame(columns=['Variable', 'Category'] +
                              [f"{group}_Mean" for group in treatment_groups] +
                              [f"{group}_SD" for group in treatment_groups] +
                              ['p-value'])

    # Continuous variables
    for var in ['age', 'baseline_vaccine_confidence']:
        means = []
        sds = []

        for group in treatment_groups:
            group_data = merged_df[merged_df['treatment_group'] == group][var]
            means.append(group_data.mean())
            sds.append(group_data.std())

        # Perform ANOVA
        from scipy.stats import f_oneway
        groups_data = [merged_df[merged_df['treatment_group'] == group][var] for group in treatment_groups]
        f_stat, p_val = f_oneway(*groups_data)

        row_data = {'Variable': var, 'Category': ''}
        for i, group in enumerate(treatment_groups):
            row_data[f"{group}_Mean"] = f"{means[i]:.2f}"
            row_data[f"{group}_SD"] = f"{sds[i]:.2f}"
        row_data['p-value'] = format_p_value(p_val)

        balance_table = pd.concat([balance_table, pd.DataFrame([row_data])], ignore_index=True)

    # Categorical variables
    for var in ['gender', 'region', 'education', 'income', 'political_leaning']:
        categories = merged_df[var].unique()

        for category in categories:
            percentages = []
            counts = []

            for group in treatment_groups:
                group_data = merged_df[merged_df['treatment_group'] == group]
                category_count = sum(group_data[var] == category)
                category_pct = category_count / len(group_data) * 100
                percentages.append(category_pct)
                counts.append(category_count)

            # Chi-square test
            from scipy.stats import chi2_contingency
            contingency_table = pd.crosstab(merged_df[var] == category, merged_df['treatment_group'])
            chi2, p_val, _, _ = chi2_contingency(contingency_table)

            row_data = {'Variable': var, 'Category': category}
            for i, group in enumerate(treatment_groups):
                row_data[f"{group}_Mean"] = f"{percentages[i]:.1f}%"
                row_data[f"{group}_SD"] = f"({counts[i]})"
            row_data['p-value'] = format_p_value(p_val)

            balance_table = pd.concat([balance_table, pd.DataFrame([row_data])], ignore_index=True)

    balance_table.to_csv('tables/balance_table.csv', index=False)

    return balance_table

def create_attrition_table(attrition_analysis, merged_df):
    # Get counts of completers and non-completers
    n_completers = sum(merged_df['completed_endline'] == 1)
    n_non_completers = sum(merged_df['completed_endline'] == 0)

    attrition_table = pd.DataFrame(columns=['Variable', 'Category',
                                         'Completers', 'Non-completers',
                                         'Test', 'Statistic', 'p-value'])
    # Continuous variables
    for var in ['age', 'baseline_vaccine_confidence']:
        completers_mean = merged_df[merged_df['completed_endline'] == 1][var].mean()
        completers_sd = merged_df[merged_df['completed_endline'] == 1][var].std()
        non_completers_mean = merged_df[merged_df['completed_endline'] == 0][var].mean()
        non_completers_sd = merged_df[merged_df['completed_endline'] == 0][var].std()

        test_row = attrition_analysis[attrition_analysis['Variable'] == var].iloc[0]

        new_row = pd.DataFrame([{
            'Variable': var,
            'Category': '',
            'Completers': f"{completers_mean:.2f} ({completers_sd:.2f})",
            'Non-completers': f"{non_completers_mean:.2f} ({non_completers_sd:.2f})",
            'Test': test_row['Test'],
            'Statistic': f"{test_row['Statistic']:.3f}",
            'p-value': format_p_value(test_row['p_value'])
        }])
        attrition_table = pd.concat([attrition_table, new_row], ignore_index=True)

    # Categorical variables
    for var in ['gender', 'region', 'education', 'income', 'political_leaning', 'treatment_group']:
        categories = merged_df[var].unique()

        for category in categories:
            completers_count = sum((merged_df['completed_endline'] == 1) & (merged_df[var] == category))
            completers_pct = completers_count / n_completers * 100

            non_completers_count = sum((merged_df['completed_endline'] == 0) & (merged_df[var] == category))
            non_completers_pct = non_completers_count / n_non_completers * 100

            test_row = attrition_analysis[attrition_analysis['Variable'] == var].iloc[0]

            new_row = pd.DataFrame([{
                'Variable': var,
                'Category': category,
                'Completers': f"{completers_pct:.1f}% ({completers_count})",
                'Non-completers': f"{non_completers_pct:.1f}% ({non_completers_count})",
                'Test': test_row['Test'],
                'Statistic': f"{test_row['Statistic']:.3f}",
                'p-value': format_p_value(test_row['p_value'])
            }])
            attrition_table = pd.concat([attrition_table, new_row], ignore_index=True)

    attrition_table.to_csv('tables/attrition_table.csv', index=False)

    return attrition_table

def create_treatment_effects_table(treatment_effects):
    base_model = treatment_effects['base_model']
    adjusted_model = treatment_effects['adjusted_model']

    # Create table structure
    effects_table = pd.DataFrame(columns=['Model', 'Treatment', 'Coefficient', 'SE', 't-value', 'p-value', '95% CI'])

    # extract model coefficients
    def extract_model_coefs(model, model_name):
        if isinstance(model, str):
            return []

        reason_coef = model.params.get('C(treatment_group, Treatment("control"))[T.reason]')
        reason_se = model.bse.get('C(treatment_group, Treatment("control"))[T.reason]')
        reason_tval = model.tvalues.get('C(treatment_group, Treatment("control"))[T.reason]')
        reason_pval = model.pvalues.get('C(treatment_group, Treatment("control"))[T.reason]')
        reason_ci = model.conf_int().loc['C(treatment_group, Treatment("control"))[T.reason]']

        emotion_coef = model.params.get('C(treatment_group, Treatment("control"))[T.emotion]')
        emotion_se = model.bse.get('C(treatment_group, Treatment("control"))[T.emotion]')
        emotion_tval = model.tvalues.get('C(treatment_group, Treatment("control"))[T.emotion]')
        emotion_pval = model.pvalues.get('C(treatment_group, Treatment("control"))[T.emotion]')
        emotion_ci = model.conf_int().loc['C(treatment_group, Treatment("control"))[T.emotion]']

        rows = []
        if reason_coef is not None:
            rows.append({
                'Model': model_name,
                'Treatment': 'Reason',
                'Coefficient': f"{reason_coef:.3f}",
                'SE': f"{reason_se:.3f}",
                't-value': f"{reason_tval:.3f}",
                'p-value': format_p_value(reason_pval),
                '95% CI': f"({reason_ci[0]:.3f}, {reason_ci[1]:.3f})"
            })

        if emotion_coef is not None:
            rows.append({
                'Model': model_name,
                'Treatment': 'Emotion',
                'Coefficient': f"{emotion_coef:.3f}",
                'SE': f"{emotion_se:.3f}",
                't-value': f"{emotion_tval:.3f}",
                'p-value': format_p_value(emotion_pval),
                '95% CI': f"({emotion_ci[0]:.3f}, {emotion_ci[1]:.3f})"
            })

        return rows

    # base model coefficients
    for row in extract_model_coefs(base_model, "Unadjusted"):
        effects_table = pd.concat([effects_table, pd.DataFrame([row])], ignore_index=True)

    # adjusted model coefficients
    for row in extract_model_coefs(adjusted_model, "Adjusted"):
        effects_table = pd.concat([effects_table, pd.DataFrame([row])], ignore_index=True)

    effects_table.to_csv('tables/treatment_effects.csv', index=False)

    return effects_table

def create_education_effects_table(heterogeneous_effects):
    education_models = heterogeneous_effects['education_models']

    edu_table = pd.DataFrame(columns=['Education Level', 'Treatment', 'Coefficient', 'SE', 'p-value', '95% CI'])

    for edu, model in education_models.items():
        if isinstance(model, str):
            new_row = pd.DataFrame([{
                'Education Level': edu,
                'Treatment': 'Reason',
                'Coefficient': 'N/A',
                'SE': 'N/A',
                'p-value': 'N/A',
                '95% CI': model
            }])
            edu_table = pd.concat([edu_table, new_row], ignore_index=True)

            new_row = pd.DataFrame([{
                'Education Level': edu,
                'Treatment': 'Emotion',
                'Coefficient': 'N/A',
                'SE': 'N/A',
                'p-value': 'N/A',
                '95% CI': model
            }])
            edu_table = pd.concat([edu_table, new_row], ignore_index=True)
        else:
            reason_coef = model.params.get('C(treatment_group)[T.reason]')
            reason_se = model.bse.get('C(treatment_group)[T.reason]')
            reason_pval = model.pvalues.get('C(treatment_group)[T.reason]')
            reason_ci = model.conf_int().loc['C(treatment_group)[T.reason]']

            emotion_coef = model.params.get('C(treatment_group)[T.emotion]')
            emotion_se = model.bse.get('C(treatment_group)[T.emotion]')
            emotion_pval = model.pvalues.get('C(treatment_group)[T.emotion]')
            emotion_ci = model.conf_int().loc['C(treatment_group)[T.emotion]']

            if reason_coef is not None:
                new_row = pd.DataFrame([{
                    'Education Level': edu,
                    'Treatment': 'Reason',
                    'Coefficient': f"{reason_coef:.3f}",
                    'SE': f"{reason_se:.3f}",
                    'p-value': format_p_value(reason_pval),
                    '95% CI': f"({reason_ci[0]:.3f}, {reason_ci[1]:.3f})"
                }])
                edu_table = pd.concat([edu_table, new_row], ignore_index=True)

            if emotion_coef is not None:
                new_row = pd.DataFrame([{
                    'Education Level': edu,
                    'Treatment': 'Emotion',
                    'Coefficient': f"{emotion_coef:.3f}",
                    'SE': f"{emotion_se:.3f}",
                    'p-value': format_p_value(emotion_pval),
                    '95% CI': f"({emotion_ci[0]:.3f}, {emotion_ci[1]:.3f})"
                }])
                edu_table = pd.concat([edu_table, new_row], ignore_index=True)

    edu_table.to_csv('tables/education_effects.csv', index=False)

    return edu_table

def create_all_tables(results, merged_df):
    setup_table_folder()
    create_demographics_table(results['descriptive_stats'])
    create_balance_table(merged_df)
    create_attrition_table(results['attrition_analysis'], merged_df)
    create_treatment_effects_table(results['treatment_effects'])
    create_education_effects_table(results['heterogeneous_effects'])

    print("tables created")

if __name__ == "__main__":
    print("Running analysis...")
    results = run_analysis()

    # Get merged dataframe for some table calculations
    baseline_df, treatment_df, endline_df = load_data()
    merged_df = merge_datasets(baseline_df, treatment_df, endline_df)

    print("Creating tables...")
    create_all_tables(results, merged_df)
