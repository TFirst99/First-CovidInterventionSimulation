import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_merged_data(data_dir='data'):
    # Load merged dataset
    merged_df = pd.read_csv(f'{data_dir}/merged_data.csv')
    return merged_df

def create_visualizations(merged_df, output_dir='results/figures'):
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Set style
    sns.set_style("whitegrid")

    # 1. Vaccine uptake by treatment group
    plt.figure(figsize=(10, 6))
    uptake_by_group = merged_df.groupby('treatment_group')['vaccine_uptake'].mean()
    ax = sns.barplot(x=uptake_by_group.index, y=uptake_by_group.values)

    # Add value labels on top of bars
    for i, v in enumerate(uptake_by_group.values):
        ax.text(i, v + 0.01, f'{v:.3f}', ha='center')

    plt.title('COVID-19 Vaccine Uptake by Treatment Group')
    plt.ylabel('Proportion Vaccinated')
    plt.xlabel('Treatment Group')
    plt.savefig(f'{output_dir}/vaccine_uptake_by_treatment.png', dpi=300, bbox_inches='tight')

    # 2. Follow-up intention by treatment group (for those who responded)
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='treatment_group', y='follow_up_intention', data=merged_df.dropna(subset=['follow_up_intention']))
    plt.title('Follow-up Vaccine Intention by Treatment Group')
    plt.ylabel('Vaccine Intention (1-7 scale)')
    plt.xlabel('Treatment Group')
    plt.savefig(f'{output_dir}/intention_by_treatment.png', dpi=300, bbox_inches='tight')

    # 3. Vaccine uptake by demographic factors
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Age groups
    merged_df['age_group'] = pd.cut(merged_df['age'], bins=[17, 30, 45, 60, 85],
                                    labels=['18-30', '31-45', '46-60', '61+'])

    sns.barplot(x='age_group', y='vaccine_uptake', hue='treatment_group', data=merged_df, ax=axes[0, 0])
    axes[0, 0].set_title('Vaccine Uptake by Age Group and Treatment')
    axes[0, 0].set_xlabel('Age Group')
    axes[0, 0].set_ylabel('Proportion Vaccinated')

    # Gender
    sns.barplot(x='gender', y='vaccine_uptake', hue='treatment_group', data=merged_df, ax=axes[0, 1])
    axes[0, 1].set_title('Vaccine Uptake by Gender and Treatment')
    axes[0, 1].set_xlabel('Gender')
    axes[0, 1].set_ylabel('Proportion Vaccinated')

    # Region
    sns.barplot(x='region', y='vaccine_uptake', hue='treatment_group', data=merged_df, ax=axes[1, 0])
    axes[1, 0].set_title('Vaccine Uptake by Region and Treatment')
    axes[1, 0].set_xlabel('Region')
    axes[1, 0].set_ylabel('Proportion Vaccinated')

    # Education
    sns.barplot(x='education', y='vaccine_uptake', hue='treatment_group', data=merged_df, ax=axes[1, 1])
    axes[1, 1].set_title('Vaccine Uptake by Education and Treatment')
    axes[1, 1].set_xlabel('Education Level')
    axes[1, 1].set_ylabel('Proportion Vaccinated')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/uptake_by_demographics.png', dpi=300, bbox_inches='tight')

    print(f"Visualizations saved to {output_dir}/")

if __name__ == "__main__":
    merged_df = load_merged_data()
    create_visualizations(merged_df)
