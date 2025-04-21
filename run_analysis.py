import os
import argparse
from generate_data import generate_baseline_data, generate_treatment_assignment, generate_endline_data, merge_datasets, save_datasets
from analyze_data import load_data, run_analysis

def setup_directories():
    print("Creating directories")
    directories = ['data', 'figures', 'tables']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def run_pipeline(seed=42):
    print("Generating data")
    baseline_df = generate_baseline_data(seed=seed)
    treatment_df = generate_treatment_assignment(baseline_df, seed=seed)
    endline_df = generate_endline_data(baseline_df, treatment_df, seed=seed, response_rate=0.9)
    merged_df = merge_datasets(baseline_df, treatment_df, endline_df)
    save_datasets(baseline_df, treatment_df, endline_df, merged_df)

    print("Analyzing data")
    results = run_analysis()

    print("Creating visualizations")
    from create_figures import create_all_figures
    create_all_figures(results)

    print("Creating tables")
    from create_tables import create_all_tables
    baseline_df, treatment_df, endline_df, merged_df = load_data()
    create_all_tables(results, merged_df)

    print("Pipeline completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='COVID-19 Vaccine Ad Experiment Simulation')
    parser.add_argument('--seed', type=int, default=42, help='seed for random generation')
    args = parser.parse_args()

    setup_directories()
    run_pipeline(seed=args.seed)
