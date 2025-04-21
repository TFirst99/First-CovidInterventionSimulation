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
4. `merged_data.csv`: merged dataset of results

## How to Run the Pipeline
1. Create and activate a virtual environment (recommended):
```
python -m venv venv
source venv/bin/activate
```
On Windows, use:
```
venv\Scripts\activate
```

2. Make sure you have all dependencies installed:
```
pip install -r requirements.txt
```

3. Run the entire pipeline:
```
python run_analysis.py
```

4. Alternatively, run individual components:
   - Data generation: `python generate_data.py`
   - Data analysis: `python analyze_data.py`
   - Create figures: `python create_figures.py`
   - Create tables: `python create_tables.py`

## Output
- `data/`: Contains the generated datasets
- `figures/`: Contains visualization figures
- `tables/`: Contains statistical tables

## Methodological Details
### Data Generation

- **Demographics**: Baseline participant characteristics include age, gender, region, political leaning, education level, and income level. More complex characteristics (political leaning, education, income) are generated with realistic correlations based on primary demographics.

- **Baseline Attitudes**: Initial vaccine confidence is measured on a 1-7 scale, influenced by education level and political leaning,

### Treatment Assignment
- Participants are randomly assigned to one of three conditions with equal probability (1/3 each):
  - Reason-based ad campaign
  - Emotion-based ad campaign
  - Control group (no ads)

### Endline Data Collection
- Follow-up survey conducted with a 90% response rate (simulated attrition)
- Key outcome measures:
  - Vaccine uptake (binary: yes/no)
  - Endline vaccine confidence (1-7 scale)
- Treatment effects are modeled with realistic heterogeneity:
  - Reason-based ads have a greater effect on higher education individuals
  - Emotion-based ads have more consistent effects across education levels
  - Both treatments increase vaccine uptake compared to control

### Statistical Analysis

#### Randomization Check
- Randomization checks ensure that treatment groups are balanced on observable characteristics
- Chi-squared tests for categorical variables (gender, region, education, income, political leaning)
- ANOVA for continuous variables (age, baseline vaccine confidence)

#### Attrition Analysis
- Attrition analysis ensures that the sample completing the endline survey remains representative of the baseline random sample
- Chi-squared tests for differential attrition by categorical variables
- T-tests for differential attrition by continuous variables

#### Treatment Effect Estimation
- Primary model: Logistic regression of vaccine uptake on treatment indicators
- Adjusted model: Adds demographic controls (age, gender, education, region, political leaning, baseline confidence)
- Reports average treatment effects with confidence intervals

#### Heterogeneous Effects Analysis
- Analyzes the difference in treatment response across education levels

#### Vaccine Confidence Analysis
- Evaluates the secondary impact of the intervention on vaccine confidence
- ANCOVA models examining changes in vaccine confidence
- Within-group paired t-tests comparing baseline to endline confidence

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
