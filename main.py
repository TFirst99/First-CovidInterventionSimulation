import os
import argparse
from generate_data import generate_baseline_data, generate_treatment_assignment, generate_endline_data, save_datasets
from analyze_data import load_data, merge_datasets, analyze_effects, save_results
from visualize_data import load_merged_data, create_visualizations

def setup_directories():
    print("creating directories")
    directories = ['data', 'results/tables', 'results/figures']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def run_pipeline(seed=42):
    print("generating data")
    # generate data
    baseline_df = generate_baseline_data(seed=seed)
    treatment_df = generate_treatment_assignment(baseline_df, seed=seed)
    endline_df = generate_endline_data(baseline_df, treatment_df, seed=seed)
    save_datasets(baseline_df, treatment_df, endline_df)

    print("analyzing data")
    # analyze data
    baseline_df, treatment_df, endline_df = load_data()
    merged_df = merge_datasets(baseline_df, treatment_df, endline_df)
    results = analyze_effects(merged_df)
    summary_by_group, test_results = save_results(merged_df, results)
    print("creating visualizations")

    # create visualizations
    merged_df = load_merged_data()
    create_visualizations(merged_df)

    # Print key results
    print("\n***** RESULTS: ******")
    print("\nVaccine Uptake:")
    print(results['vaccine_rates'])

    print("\nStatistical Test Results:")
    print(test_results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='COVID-19 Vaccine Ad Experiment Simulation')
    parser.add_argument('--seed', type=int, default=42, help='seed for random generation')
    args = parser.parse_args()

    setup_directories()
    run_pipeline(seed=args.seed)
