import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import matplotlib.patches as mpatches
from analyze_data import run_analysis

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk")

COLORS = {
    'reason': '#4E79A7',
    'emotion': '#F28E2B',
    'control': '#76B7B2',
    'neutral': '#76B7B2'
}

def setup_figure_folder():
    os.makedirs('figures', exist_ok=True)

def create_political_confidence_figure(political_confidence):
    plt.figure(figsize=(10, 6))

    # Prepare data
    political_leanings = political_confidence.index
    means = political_confidence['mean']
    ci_lower = political_confidence['ci_lower']
    ci_upper = political_confidence['ci_upper']
    error = [means - ci_lower, ci_upper - means]

    # Create bar chart
    bars = plt.bar(political_leanings, means, color=COLORS['neutral'],
                   yerr=error, capsize=5, alpha=0.8)

    # Add labels and formatting
    plt.xlabel('Political Leaning')
    plt.ylabel('Baseline Vaccine Confidence (1-7 Scale)')
    plt.title('Baseline Vaccine Confidence by Political Leaning')

    # Rotate x labels if needed
    plt.xticks(rotation=0)

    plt.ylim(0, 7)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig('figures/political_confidence.png', dpi=300)
    plt.close()

def create_treatment_effect_figure(treatment_stats):
    plt.figure(figsize=(10, 6))

    # Prepare data
    groups = treatment_stats.index
    percentages = treatment_stats['percentage']
    ci_lower = treatment_stats['ci_lower'] * 100
    ci_upper = treatment_stats['ci_upper'] * 100
    error = [percentages - ci_lower, ci_upper - percentages]

    # Set bar colors based on treatment group
    colors = [COLORS[group] for group in groups]

    # Create bar chart
    bars = plt.bar(groups, percentages, color=colors,
                   yerr=error, capsize=5, alpha=0.8)

    # Add labels and formatting
    plt.xlabel('Treatment Group')
    plt.ylabel('Vaccination Rate (%)')
    plt.title('Vaccination Rates by Treatment Group')

    plt.ylim(0, 100)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig('figures/treatment_effect.png', dpi=300)
    plt.close()

def create_confidence_change_figure(confidence_stats):
    plt.figure(figsize=(10, 6))

    # Prepare data
    groups = confidence_stats.index
    x = np.arange(len(groups))
    width = 0.35

    # Get baseline and endline values
    baseline_means = confidence_stats['baseline_mean']
    endline_means = confidence_stats['endline_mean']

    # Compute error bars
    baseline_error = [baseline_means - confidence_stats['baseline_ci_lower'],
                     confidence_stats['baseline_ci_upper'] - baseline_means]
    endline_error = [endline_means - confidence_stats['endline_ci_lower'],
                    confidence_stats['endline_ci_upper'] - endline_means]

    # Set bar colors based on treatment group, with lighter shade for baseline
    baseline_colors = [adjust_lightness(COLORS[group], 1.3) for group in groups]
    endline_colors = [COLORS[group] for group in groups]

    # Create paired bar chart
    plt.bar(x - width/2, baseline_means, width, color=baseline_colors,
           yerr=baseline_error, capsize=5, label='Baseline', alpha=0.8)
    plt.bar(x + width/2, endline_means, width, color=endline_colors,
           yerr=endline_error, capsize=5, label='Endline', alpha=0.8)

    # Add labels and formatting
    plt.xlabel('Treatment Group')
    plt.ylabel('Vaccine Confidence (1-7 Scale)')
    plt.title('Change in Vaccine Confidence by Treatment Group')
    plt.xticks(x, groups)

    plt.ylim(0, 7.5)  # Scale goes from 1-7
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()

    plt.tight_layout()
    plt.savefig('figures/confidence_change.png', dpi=300)
    plt.close()

def adjust_lightness(color, amount=0.5):
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])

def create_education_effect_figure(education_stats):
    plt.figure(figsize=(12, 6))

    education_levels = education_stats.index.get_level_values('education').unique()
    treatment_groups = education_stats.index.get_level_values('treatment_group').unique()

    x = np.arange(len(education_levels))
    width = 0.25

    # Generate bars for each treatment group
    for i, group in enumerate(treatment_groups):
        means = []
        errors_lower = []
        errors_upper = []

        for edu in education_levels:
            if (edu, group) in education_stats.index:
                means.append(education_stats.loc[(edu, group), 'percentage'])
                errors_lower.append(education_stats.loc[(edu, group), 'ci_lower'] * 100)
                errors_upper.append(education_stats.loc[(edu, group), 'ci_upper'] * 100)
            else:
                means.append(0)
                errors_lower.append(0)
                errors_upper.append(0)

        yerr = [np.array(means) - np.array(errors_lower),
                np.array(errors_upper) - np.array(means)]

        plt.bar(x + (i - 1) * width, means, width, label=group.capitalize(),
               color=COLORS[group], yerr=yerr, capsize=5, alpha=0.8)

    # Add labels and formatting
    plt.xlabel('Education Level')
    plt.ylabel('Vaccination Rate (%)')
    plt.title('Treatment Effects by Education Level')
    plt.xticks(x, education_levels)

    plt.ylim(0, 100)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()

    plt.tight_layout()
    plt.savefig('figures/education_effect.png', dpi=300)
    plt.close()

def create_key_findings_figure(key_findings):
    plt.figure(figsize=(10, 6))

    key_findings = key_findings.sort_values('Comparison', key=lambda x: [
        0 if 'Reason vs' in i else
        1 if 'Emotion vs' in i else
        2 if 'Reason Effect' in i and 'Graduate' in i else
        3 for i in x
    ])

    comparisons = key_findings['Comparison']
    effects = key_findings['Effect']
    errors = [effects - key_findings['CI_Lower'], key_findings['CI_Upper'] - effects]

    colors = []
    for comp in comparisons:
        if 'Reason' in comp:
            colors.append(COLORS['reason'])
        elif 'Emotion' in comp:
            colors.append(COLORS['emotion'])
        else:
            colors.append(COLORS['neutral'])

    plt.figure(figsize=(10, 6))
    y_pos = np.arange(len(comparisons))

    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)

    bars = plt.barh(y_pos, effects, xerr=errors, capsize=5,
                  color=colors, alpha=0.8)

    plt.yticks(y_pos, comparisons)
    plt.xlabel('Effect Size (Percentage Points)')
    plt.title('Summary of Key Findings')

    plt.grid(axis='x', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig('figures/key_findings.png', dpi=300)
    plt.close()

def create_all_figures(results):
    setup_figure_folder()
    create_political_confidence_figure(results['political_confidence'])
    create_treatment_effect_figure(results['treatment_effects']['treatment_stats'])
    create_confidence_change_figure(results['confidence_analysis']['confidence_stats'])
    create_education_effect_figure(results['heterogeneous_effects']['education_stats'])
    create_key_findings_figure(results['key_findings'])

if __name__ == "__main__":
    print("Running analysis...")
    results = run_analysis()

    print("Creating figures...")
    create_all_figures(results)
