## Overview
This project simulates and analyzes data for a field experiment evaluating the effectiveness of different Facebook ad campaigns on COVID-19 vaccine uptake. The experiment involves 5,000 participants across the US randomly assigned to one of three conditions: a reason-based ad campaign, an emotion-based ad campaign, or a control group (no ads).

## Project Structure
- `generate_data.py`: Creates synthetic data for the experiment
- `analyze_data.py`: Performs statistical analysis on the data
- `create_figures.py`: Generates visualizations
- `create_tables.py`: Creates csv tables summarizing results
- `run.py`: Runs the entire pipeline

## Data Description
The simulation creates three primary datasets:
1. `baseline_survey.csv`: Demographics and initial attitudes
2. `treatment_assignment.csv`: Random assignment to treatment groups
3. `endline_survey.csv`: Outcomes after exposure (completed by 90% of participants)

## How to Run the Pipeline
1. Make sure you have all dependencies installed:
   ```
  pip install -r requirements.txt
   ```

2. Run the entire pipeline:
   ```
   python run_pipeline.py
   ```

3. Alternatively, run individual components:
   - Data generation: `python generate_data.py`
   - Data analysis: `python analyze_data.py`
   - Create figures: `python create_figures.py`
   - Create tables: `python create_tables.py`

## Output
- `data/`: Contains the generated datasets
- `figures/`: Contains visualization figures
- `tables/`: Contains statistical tables

## Methodological Details
- **Model Specification**: Logistic regression models assess the causal effect of ad campaigns on vaccination rates
- **Randomization Check**: Chi-squared tests and ANOVA verify successful randomization
- **Attrition Analysis**: Examines whether participants who dropped out differ from those who completed the study
- **Heterogeneous Effects**: Investigates how treatment effects vary by education level

## Visualizations
- **Fig 1**: Baseline Vaccine Confidence by Political Leaning
- **Fig 2**: Vaccination Rates by Treatment Group
- **Fig 3**: Change in Vaccine Confidence
- **Fig 4**: Treatment Effects by Education Level
- **Fig 5**: Summary of Key Findings

## Tables
- **Table 1**: Participant Demographics
- **Table 2**: Balance Table
- **Table 3**: Attrition Analysis
- **Table 4**: Treatment Effect Estimates
- **Table 5**: Stratified Treatment Effects by Education
