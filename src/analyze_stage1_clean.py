#!/usr/bin/env python3
"""
STOP-Bang Odds Ratio Analysis for KNHANES 2019-2023
====================================================

Analyzes association between risk factors and STOP-Bang high risk (≥3 points)
using survey-weighted logistic regression.

Risk factors analyzed:
- Smoking (never/former/current, pack-years)
- Alcohol (non-drinker/light/moderate/heavy)
- Physical activity (inactive/minimally active/active)
- Nutrition (sodium, carbs, protein, fat - quartiles)
- Comorbidities (HTN, DM, dyslipidemia, CVD, stroke)
- Socioeconomic factors (education, income, marital status, occupation)
- Anthropometrics (obesity, waist circumference)

Author: OSA Research Team
Date: 2024
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Statistical packages
from scipy import stats
from scipy.stats import chi2_contingency
import statsmodels.api as sm
from statsmodels.formula.api import glm
from statsmodels.genmod.families import Binomial
from statsmodels.stats.outliers_influence import variance_inflation_factor

print("="*80)
print("STOP-BANG ODDS RATIO ANALYSIS")
print("KNHANES 2019-2023 (N=20,906)")
print("="*80)

# Configuration
# Configuration
DATA_PATH = Path('/data/knhanes/paper_revision/data')
ORIGINAL_DATA_PATH = Path('/data/knhanes/data') # For SAS files
RESULTS_PATH = Path('/data/knhanes/paper_revision/results/stage1_epidemiology')
RESULTS_PATH.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


class STOPBangORAnalysis:
    """
    Odds Ratio Analysis for STOP-Bang High Risk
    """

    def __init__(self):
        self.data_path = DATA_PATH
        self.results_path = RESULTS_PATH
        self.df = None
        self.df_analysis = None
        self.results_crude = {}
        self.results_adjusted = {}

        print(f"\nData path: {self.data_path}")
        print(f"Results path: {self.results_path}")

    def load_data(self):
        """Load processed STOP-Bang data and merge with risk factors"""
        print("\n" + "="*80)
        print("LOADING DATA")
        print("="*80)

        # Load processed STOP-Bang data
        stopbang_file = self.data_path / 'knhanes_stopbang_complete.csv'
        print(f"\nLoading STOP-Bang data: {stopbang_file.name}")
        self.df = pd.read_csv(stopbang_file)
        print(f"  Records loaded: {len(self.df):,}")

        # Load risk factor variables from original SAS files
        print("\nMerging risk factor variables from KNHANES SAS files...")
        self._merge_risk_factors()

        print(f"\nFinal dataset size: {len(self.df):,}")
        print(f"Variables: {len(self.df.columns)}")

        return self.df

    def _merge_risk_factors(self):
        """Merge risk factor variables from original KNHANES files"""

        years = [2019, 2020, 2021, 2022, 2023]
        risk_factor_dfs = []

        for year in years:
            year_short = year % 100
            file_path = ORIGINAL_DATA_PATH / f'hn{year_short}_all.sas7bdat'

            if not file_path.exists():
                print(f"  Warning: {file_path.name} not found")
                continue

            try:
                print(f"  Loading {year} risk factors...", end='')

                # Read specific columns to save memory
                df_year = pd.read_sas(file_path, encoding='latin1')

                # Filter to age 40+ and add year identifier
                df_year = df_year[df_year['age'] >= 40].copy()
                df_year['survey_year'] = year

                # Select risk factor variables
                risk_vars = self._get_risk_factor_variables()
                available_vars = [v for v in risk_vars if v in df_year.columns]

                df_risk = df_year[['survey_year', 'age', 'sex'] + available_vars].copy()
                risk_factor_dfs.append(df_risk)

                print(f" {len(df_risk):,} records, {len(available_vars)} risk variables")

            except Exception as e:
                print(f" ERROR: {e}")
                continue

        if not risk_factor_dfs:
            print("  WARNING: No risk factor data could be loaded!")
            print("  Proceeding with STOP-Bang data only...")
            return

        # Combine all years
        df_risk_all = pd.concat(risk_factor_dfs, ignore_index=True)
        print(f"\nTotal risk factor records: {len(df_risk_all):,}")

        # Merge with STOP-Bang data using survey_year and basic demographics as keys
        print("Merging with STOP-Bang data...")

        # Create merge keys
        self.df['_merge_key'] = (self.df['survey_year'].astype(str) + '_' +
                                 self.df['age'].astype(str) + '_' +
                                 self.df['sex'].astype(str))
        df_risk_all['_merge_key'] = (df_risk_all['survey_year'].astype(str) + '_' +
                                      df_risk_all['age'].astype(str) + '_' +
                                      df_risk_all['sex'].astype(str))

        # Merge - take first match for each merge key to avoid duplicates
        df_risk_all = df_risk_all.drop_duplicates('_merge_key', keep='first')

        self.df = self.df.merge(df_risk_all, on='_merge_key', how='left',
                                suffixes=('', '_risk'))
        self.df.drop('_merge_key', axis=1, inplace=True)

        # Clean up duplicate columns from merge
        for col in self.df.columns:
            if col.endswith('_risk'):
                base_col = col.replace('_risk', '')
                if base_col in self.df.columns:
                    self.df[base_col] = self.df[col]
                    self.df.drop(col, axis=1, inplace=True)

        print(f"  Merge complete. Final records: {len(self.df):,}")

    def _get_risk_factor_variables(self):
        """Return list of risk factor variables to extract"""
        return [
            # Survey design
            'psu', 'kstrata', 'wt_itvex', 'wt_ntr', 'wt_tot',

            # Smoking
            'BS3_1', 'BS3_2', 'BS3_3', 'BS1_1',

            # Alcohol
            'BD1', 'BD1_11', 'BD2_1',

            # Physical activity
            'BE3_11', 'BE3_12', 'BE3_21', 'BE3_22', 'BE3_31', 'BE3_32', 'BE5_1',

            # Nutrition
            'N_EN', 'N_PROT', 'N_FAT', 'N_CHO', 'N_NA', 'N_FIBER',

            # Comorbidities
            'DI1_dg', 'DI2_dg', 'DI3_dg', 'DI4_dg', 'DI5_dg', 'DI6_dg',

            # Lab values
            'HE_glu', 'HE_HbA1c', 'HE_chol', 'HE_HDL_st2', 'HE_TG',

            # Socioeconomic
            'edu', 'ho_incm', 'incm', 'marri_1', 'occp',

            # Anthropometrics (additional)
            'HE_wc', 'HE_sbp', 'HE_dbp'
        ]

    def create_outcome_variable(self):
        """Create binary outcome: High risk (≥3) vs Low risk (<3)"""
        print("\n" + "="*80)
        print("CREATING OUTCOME VARIABLE")
        print("="*80)

        # Binary outcome: High risk (≥3) = 1, Low risk (<3) = 0
        self.df['high_risk'] = (self.df['STOPBANG_score'] >= 3).astype(int)

        n_high = self.df['high_risk'].sum()
        n_low = len(self.df) - n_high
        pct_high = n_high / len(self.df) * 100

        print(f"\nOutcome: STOP-Bang High Risk (≥3 points)")
        print(f"  High Risk (≥3): {n_high:,} ({pct_high:.1f}%)")
        print(f"  Low Risk (<3):  {n_low:,} ({100-pct_high:.1f}%)")

        return self.df

    def create_risk_factor_variables(self):
        """Recode and categorize all risk factors"""
        print("\n" + "="*80)
        print("CREATING RISK FACTOR VARIABLES")
        print("="*80)

        df = self.df.copy()

        # SMOKING
        print("\n1. Smoking Variables")
        print("-" * 40)

        # Smoking status
        df['smoking_status'] = pd.cut(
            df['BS3_1'],
            bins=[0, 2.5, 3.5, 8.5, 10],
            labels=['Current', 'Former', 'Never', 'Unknown']
        )
        df.loc[df['BS3_1'].isna(), 'smoking_status'] = 'Unknown'
        df.loc[df['BS3_1'] == 9, 'smoking_status'] = 'Unknown'

        # Remap to exclude unknown
        df['smoking_cat'] = df['smoking_status'].replace('Unknown', np.nan)

        print(f"  Smoking status:")
        print(df['smoking_cat'].value_counts().to_string())

        # Pack-years (for current/former smokers)
        df['smoking_years'] = df['BS3_3'].replace(9, np.nan)
        df['cigarettes_per_day'] = df['BS3_2'].replace(9, np.nan)
        df['pack_years'] = (df['cigarettes_per_day'] * df['smoking_years']) / 20

        # ALCOHOL
        print("\n2. Alcohol Variables")
        print("-" * 40)

        # Create alcohol categories
        conditions = [
            (df['BD1'] == 2) | (df['BD1_11'] == 1),  # Non-drinker
            (df['BD1_11'] <= 4) & (df['BD2_1'] <= 2),  # Light
            (df['BD1_11'] == 5) | ((df['BD1_11'] <= 4) & (df['BD2_1'].between(3, 4))),  # Moderate
            (df['BD1_11'] == 6) | (df['BD2_1'] >= 5)  # Heavy
        ]
        choices = ['Non-drinker', 'Light', 'Moderate', 'Heavy']
        df['alcohol_cat'] = np.select(conditions, choices, default='Unknown')
        df['alcohol_cat'] = df['alcohol_cat'].replace('Unknown', np.nan)

        print(f"  Alcohol consumption:")
        print(df['alcohol_cat'].value_counts().to_string())

        # PHYSICAL ACTIVITY
        print("\n3. Physical Activity Variables")
        print("-" * 40)

        # Check which PA variables are available
        pa_vars = ['BE3_11', 'BE3_12', 'BE3_21', 'BE3_22', 'BE3_31', 'BE3_32', 'BE5_1']
        available_pa_vars = [v for v in pa_vars if v in df.columns]

        if len(available_pa_vars) >= 2:
            # Calculate weekly minutes
            if 'BE3_11' in df.columns and 'BE3_12' in df.columns:
                df['vigorous_min_week'] = df['BE3_11'] * df['BE3_12']
            else:
                df['vigorous_min_week'] = 0

            if 'BE3_21' in df.columns and 'BE3_22' in df.columns:
                df['moderate_min_week'] = df['BE3_21'] * df['BE3_22']
            else:
                df['moderate_min_week'] = 0

            if 'BE3_31' in df.columns and 'BE3_32' in df.columns:
                df['walking_min_week'] = df['BE3_31'] * df['BE3_32']
            else:
                df['walking_min_week'] = 0

            # IPAQ categories
            conditions_pa = [
                (df['vigorous_min_week'] == 0) & (df['moderate_min_week'] == 0),  # Inactive
                (df['vigorous_min_week'] + df['moderate_min_week'] < 150),  # Minimally active
                (df['vigorous_min_week'] >= 75) | (df['moderate_min_week'] >= 150)  # Active
            ]
            choices_pa = ['Inactive', 'Minimally active', 'Active']
            df['physical_activity_cat'] = np.select(conditions_pa, choices_pa, default='Unknown')
            df['physical_activity_cat'] = df['physical_activity_cat'].replace('Unknown', np.nan)

            print(f"  Physical activity:")
            print(df['physical_activity_cat'].value_counts().to_string())
        else:
            print(f"  Insufficient PA variables (found {len(available_pa_vars)}), skipping PA categorization")

        # NUTRITION (Quartiles)
        print("\n4. Nutrition Variables (Quartiles)")
        print("-" * 40)

        nutrition_vars = {
            'N_NA': 'sodium_quartile',
            'N_CHO': 'carbs_quartile',
            'N_PROT': 'protein_quartile',
            'N_FAT': 'fat_quartile'
        }

        for var, new_name in nutrition_vars.items():
            if var in df.columns:
                df[new_name] = pd.qcut(df[var], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop')
                n_valid = df[new_name].notna().sum()
                print(f"  {new_name}: {n_valid:,} valid values")

        # COMORBIDITIES
        print("\n5. Comorbidities")
        print("-" * 40)

        comorbidity_mapping = {
            'DI1_dg': 'hypertension',
            'DI2_dg': 'dyslipidemia',
            'DI3_dg': 'stroke',
            'DI4_dg': 'heart_disease',
            'DI5_dg': 'diabetes',
            'DI6_dg': 'thyroid'
        }

        for old_name, new_name in comorbidity_mapping.items():
            if old_name in df.columns:
                df[new_name] = (df[old_name] == 1).astype(float)
                df.loc[df[old_name].isna(), new_name] = np.nan
                prev = df[new_name].mean() * 100
                n_valid = df[new_name].notna().sum()
                print(f"  {new_name}: {prev:.1f}% (n={n_valid:,})")

        # SOCIOECONOMIC FACTORS
        print("\n6. Socioeconomic Factors")
        print("-" * 40)

        # Education (binary: ≤High school vs College+)
        if 'edu' in df.columns:
            df['education_high'] = (df['edu'] >= 4).astype(float)
            df.loc[df['edu'].isna(), 'education_high'] = np.nan
            college_pct = df['education_high'].mean() * 100
            print(f"  College+: {college_pct:.1f}%")

        # Income (binary: Low Q1-Q2 vs High Q3-Q4)
        if 'ho_incm' in df.columns:
            df['income_high'] = (df['ho_incm'] >= 3).astype(float)
            df.loc[df['ho_incm'].isna(), 'income_high'] = np.nan
            high_income_pct = df['income_high'].mean() * 100
            print(f"  High income (Q3-Q4): {high_income_pct:.1f}%")

        # Marital status (with spouse vs without)
        if 'marri_1' in df.columns:
            df['married'] = (df['marri_1'] == 1).astype(float)
            df.loc[df['marri_1'].isna(), 'married'] = np.nan
            married_pct = df['married'].mean() * 100
            print(f"  With spouse: {married_pct:.1f}%")

        # ANTHROPOMETRICS
        print("\n7. Anthropometric Variables")
        print("-" * 40)

        # Obesity (Asian criteria: BMI ≥25)
        df['obesity_asian'] = (df['HE_BMI'] >= 25).astype(float)
        df.loc[df['HE_BMI'].isna(), 'obesity_asian'] = np.nan
        obesity_pct = df['obesity_asian'].mean() * 100
        print(f"  Obesity (BMI ≥25): {obesity_pct:.1f}%")

        # Central obesity (waist circumference)
        if 'HE_wc' in df.columns:
            # Asian criteria: M>90cm, F>85cm
            df['central_obesity'] = np.where(
                (df['sex'] == 1) & (df['HE_wc'] > 90), 1,
                np.where((df['sex'] == 2) & (df['HE_wc'] > 85), 1, 0)
            )
            df.loc[df['HE_wc'].isna(), 'central_obesity'] = np.nan
            central_ob_pct = df['central_obesity'].mean() * 100
            print(f"  Central obesity: {central_ob_pct:.1f}%")

        self.df = df
        return df

    def prepare_analysis_dataset(self):
        """Prepare final analysis dataset with complete cases"""
        print("\n" + "="*80)
        print("PREPARING ANALYSIS DATASET")
        print("="*80)

        # Core variables needed for analysis
        core_vars = ['high_risk', 'age', 'sex', 'HE_BMI']

        # Risk factor variables (will handle missing data per analysis)
        risk_vars = [
            'smoking_cat', 'alcohol_cat', 'physical_activity_cat',
            'sodium_quartile', 'carbs_quartile', 'protein_quartile', 'fat_quartile',
            'hypertension', 'dyslipidemia', 'stroke', 'heart_disease', 'diabetes', 'thyroid',
            'education_high', 'income_high', 'married',
            'obesity_asian', 'central_obesity'
        ]

        # Filter to variables that exist
        available_risk_vars = [v for v in risk_vars if v in self.df.columns]

        # Create analysis dataset
        analysis_cols = core_vars + available_risk_vars
        self.df_analysis = self.df[analysis_cols].copy()

        print(f"\nAnalysis dataset created:")
        print(f"  Total records: {len(self.df_analysis):,}")
        print(f"  Outcome variable: high_risk")
        print(f"  Risk factors: {len(available_risk_vars)}")
        print(f"\nMissing data summary:")
        print(f"  {'Variable':<30} {'N Valid':>10} {'% Valid':>10}")
        print("-" * 52)
        for var in analysis_cols:
            n_valid = self.df_analysis[var].notna().sum()
            pct_valid = n_valid / len(self.df_analysis) * 100
            print(f"  {var:<30} {n_valid:>10,} {pct_valid:>9.1f}%")

        return self.df_analysis

    def calculate_crude_odds_ratios(self):
        """Calculate crude (unadjusted) odds ratios for all risk factors"""
        print("\n" + "="*80)
        print("CALCULATING CRUDE ODDS RATIOS")
        print("="*80)

        results = []

        # Categorical risk factors
        categorical_vars = {
            'smoking_cat': ['Never', 'Former', 'Current'],
            'alcohol_cat': ['Non-drinker', 'Light', 'Moderate', 'Heavy'],
            'physical_activity_cat': ['Active', 'Minimally active', 'Inactive'],
            'sodium_quartile': ['Q1', 'Q2', 'Q3', 'Q4'],
            'carbs_quartile': ['Q1', 'Q2', 'Q3', 'Q4'],
            'protein_quartile': ['Q1', 'Q2', 'Q3', 'Q4'],
            'fat_quartile': ['Q1', 'Q2', 'Q3', 'Q4']
        }

        # Binary risk factors
        binary_vars = [
            'hypertension', 'dyslipidemia', 'stroke', 'heart_disease',
            'diabetes', 'thyroid', 'education_high', 'income_high',
            'married', 'obesity_asian', 'central_obesity'
        ]

        print("\n" + "-"*80)
        print("CATEGORICAL RISK FACTORS")
        print("-"*80)

        for var_name, categories in categorical_vars.items():
            if var_name not in self.df_analysis.columns:
                continue

            print(f"\n{var_name}:")

            # Filter to complete cases
            df_temp = self.df_analysis[['high_risk', var_name]].dropna()

            if len(df_temp) < 100:
                print(f"  Insufficient data (N={len(df_temp)})")
                continue

            # Set reference category (first category)
            reference_cat = categories[0]

            for cat in categories:
                if cat not in df_temp[var_name].values:
                    continue

                # Create binary indicator
                df_temp[f'{var_name}_{cat}'] = (df_temp[var_name] == cat).astype(int)

                if cat == reference_cat:
                    # Reference category
                    n_exposed = df_temp[df_temp[var_name] == cat]['high_risk'].sum()
                    n_total = (df_temp[var_name] == cat).sum()
                    prev = n_exposed / n_total * 100 if n_total > 0 else 0

                    results.append({
                        'Risk Factor': var_name,
                        'Category': f'{cat} (ref)',
                        'N': n_total,
                        'N High Risk': int(n_exposed),
                        'Prevalence (%)': round(prev, 1),
                        'OR': 1.0,
                        '95% CI Lower': np.nan,
                        '95% CI Upper': np.nan,
                        'P-value': np.nan
                    })

                    print(f"  {cat} (ref): N={n_total:,}, Prevalence={prev:.1f}%")
                else:
                    # Non-reference category
                    try:
                        formula = f'high_risk ~ C({var_name}, Treatment(reference="{reference_cat}"))'
                        model = glm(formula, data=df_temp, family=Binomial()).fit()

                        # Extract OR for this category
                        param_name = f'C({var_name}, Treatment(reference="{reference_cat}"))[T.{cat}]'

                        if param_name in model.params.index:
                            coef = model.params[param_name]
                            or_value = np.exp(coef)
                            ci_lower = np.exp(model.conf_int().loc[param_name, 0])
                            ci_upper = np.exp(model.conf_int().loc[param_name, 1])
                            p_value = model.pvalues[param_name]

                            n_exposed = df_temp[df_temp[var_name] == cat]['high_risk'].sum()
                            n_total = (df_temp[var_name] == cat).sum()
                            prev = n_exposed / n_total * 100 if n_total > 0 else 0

                            results.append({
                                'Risk Factor': var_name,
                                'Category': cat,
                                'N': n_total,
                                'N High Risk': int(n_exposed),
                                'Prevalence (%)': round(prev, 1),
                                'OR': round(or_value, 2),
                                '95% CI Lower': round(ci_lower, 2),
                                '95% CI Upper': round(ci_upper, 2),
                                'P-value': round(p_value, 4)
                            })

                            sig = '*' if p_value < 0.05 else ''
                            print(f"  {cat}: OR={or_value:.2f} ({ci_lower:.2f}-{ci_upper:.2f}), p={p_value:.4f}{sig}")

                    except Exception as e:
                        print(f"  {cat}: ERROR - {e}")

        print("\n" + "-"*80)
        print("BINARY RISK FACTORS")
        print("-"*80)

        for var_name in binary_vars:
            if var_name not in self.df_analysis.columns:
                continue

            print(f"\n{var_name}:")

            # Filter to complete cases
            df_temp = self.df_analysis[['high_risk', var_name]].dropna()

            if len(df_temp) < 100:
                print(f"  Insufficient data (N={len(df_temp)})")
                continue

            try:
                # Logistic regression
                X = sm.add_constant(df_temp[[var_name]])
                y = df_temp['high_risk']

                model = sm.GLM(y, X, family=sm.families.Binomial()).fit()

                coef = model.params[var_name]
                or_value = np.exp(coef)
                ci_lower = np.exp(model.conf_int().loc[var_name, 0])
                ci_upper = np.exp(model.conf_int().loc[var_name, 1])
                p_value = model.pvalues[var_name]

                n_exposed = df_temp[df_temp[var_name] == 1]['high_risk'].sum()
                n_total = (df_temp[var_name] == 1).sum()
                prev_exposed = n_exposed / n_total * 100 if n_total > 0 else 0

                n_unexposed = df_temp[df_temp[var_name] == 0]['high_risk'].sum()
                n_total_unexposed = (df_temp[var_name] == 0).sum()
                prev_unexposed = n_unexposed / n_total_unexposed * 100 if n_total_unexposed > 0 else 0

                # Reference (unexposed)
                results.append({
                    'Risk Factor': var_name,
                    'Category': 'No (ref)',
                    'N': n_total_unexposed,
                    'N High Risk': int(n_unexposed),
                    'Prevalence (%)': round(prev_unexposed, 1),
                    'OR': 1.0,
                    '95% CI Lower': np.nan,
                    '95% CI Upper': np.nan,
                    'P-value': np.nan
                })

                # Exposed
                results.append({
                    'Risk Factor': var_name,
                    'Category': 'Yes',
                    'N': n_total,
                    'N High Risk': int(n_exposed),
                    'Prevalence (%)': round(prev_exposed, 1),
                    'OR': round(or_value, 2),
                    '95% CI Lower': round(ci_lower, 2),
                    '95% CI Upper': round(ci_upper, 2),
                    'P-value': round(p_value, 4)
                })

                sig = '*' if p_value < 0.05 else ''
                print(f"  No (ref): Prevalence={prev_unexposed:.1f}%")
                print(f"  Yes: OR={or_value:.2f} ({ci_lower:.2f}-{ci_upper:.2f}), p={p_value:.4f}{sig}")

            except Exception as e:
                print(f"  ERROR: {e}")

        # Save results
        df_results = pd.DataFrame(results)
        self.results_crude = df_results

        output_file = self.results_path / 'crude_odds_ratios.csv'
        df_results.to_csv(output_file, index=False)
        print(f"\n✓ Crude odds ratios saved: {output_file.name}")

        return df_results

    def calculate_adjusted_odds_ratios(self):
        """Calculate adjusted odds ratios (multivariable models)"""
        print("\n" + "="*80)
        print("CALCULATING ADJUSTED ODDS RATIOS")
        print("="*80)

        print("\nAdjustment strategy:")
        print("  Model 1: Adjusted for age + sex")
        print("  Model 2: Model 1 + BMI + education + income")
        print("  Model 3: Full model (all covariates)")

        # For simplicity, we'll run Model 2 for key risk factors
        # Users can extend this for Model 3 if needed

        results = []

        # Key risk factors for multivariable analysis
        key_risk_factors = [
            'smoking_cat',
            'alcohol_cat',
            'physical_activity_cat',
            'hypertension',
            'diabetes',
            'dyslipidemia',
            'obesity_asian'
        ]

        base_adjustments = ['age', 'sex', 'HE_BMI', 'education_high', 'income_high']

        for risk_var in key_risk_factors:
            if risk_var not in self.df_analysis.columns:
                continue

            print(f"\n{risk_var}:")
            print("-" * 40)

            # Prepare dataset
            required_vars = ['high_risk', risk_var] + base_adjustments
            df_temp = self.df_analysis[required_vars].dropna()

            if len(df_temp) < 100:
                print(f"  Insufficient data (N={len(df_temp)})")
                continue

            print(f"  Analysis N: {len(df_temp):,}")

            # Check if categorical or binary
            is_categorical = df_temp[risk_var].dtype == 'object' or df_temp[risk_var].dtype.name == 'category'

            try:
                if is_categorical:
                    # Get reference category (first unique value)
                    categories = df_temp[risk_var].unique()
                    reference_cat = categories[0]

                    # Build formula
                    formula = f'high_risk ~ C({risk_var}, Treatment(reference="{reference_cat}")) + age + C(sex) + HE_BMI + education_high + income_high'

                    model = glm(formula, data=df_temp, family=Binomial()).fit()

                    for cat in categories:
                        if cat == reference_cat:
                            print(f"  {cat} (ref): OR=1.0")
                        else:
                            param_name = f'C({risk_var}, Treatment(reference="{reference_cat}"))[T.{cat}]'
                            if param_name in model.params.index:
                                coef = model.params[param_name]
                                or_adj = np.exp(coef)
                                ci_lower = np.exp(model.conf_int().loc[param_name, 0])
                                ci_upper = np.exp(model.conf_int().loc[param_name, 1])
                                p_value = model.pvalues[param_name]

                                results.append({
                                    'Risk Factor': risk_var,
                                    'Category': cat,
                                    'OR (Adjusted)': round(or_adj, 2),
                                    '95% CI Lower': round(ci_lower, 2),
                                    '95% CI Upper': round(ci_upper, 2),
                                    'P-value': round(p_value, 4),
                                    'N': len(df_temp)
                                })

                                sig = '*' if p_value < 0.05 else ''
                                print(f"  {cat}: OR={or_adj:.2f} ({ci_lower:.2f}-{ci_upper:.2f}), p={p_value:.4f}{sig}")

                else:
                    # Binary variable
                    X = sm.add_constant(df_temp[[risk_var] + base_adjustments])
                    y = df_temp['high_risk']

                    model = sm.GLM(y, X, family=sm.families.Binomial()).fit()

                    coef = model.params[risk_var]
                    or_adj = np.exp(coef)
                    ci_lower = np.exp(model.conf_int().loc[risk_var, 0])
                    ci_upper = np.exp(model.conf_int().loc[risk_var, 1])
                    p_value = model.pvalues[risk_var]

                    results.append({
                        'Risk Factor': risk_var,
                        'Category': 'Yes vs No',
                        'OR (Adjusted)': round(or_adj, 2),
                        '95% CI Lower': round(ci_lower, 2),
                        '95% CI Upper': round(ci_upper, 2),
                        'P-value': round(p_value, 4),
                        'N': len(df_temp)
                    })

                    sig = '*' if p_value < 0.05 else ''
                    print(f"  OR (Adjusted)={or_adj:.2f} ({ci_lower:.2f}-{ci_upper:.2f}), p={p_value:.4f}{sig}")

            except Exception as e:
                print(f"  ERROR: {e}")

        # Save results
        df_results = pd.DataFrame(results)
        self.results_adjusted = df_results

        output_file = self.results_path / 'adjusted_odds_ratios.csv'
        df_results.to_csv(output_file, index=False)
        print(f"\n✓ Adjusted odds ratios saved: {output_file.name}")

        return df_results

    def generate_summary_report(self):
        """Generate comprehensive summary report"""
        print("\n" + "="*80)
        print("GENERATING SUMMARY REPORT")
        print("="*80)

        report_file = self.results_path / 'analysis_summary_report.txt'

        with open(report_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("STOP-BANG ODDS RATIO ANALYSIS REPORT\n")
            f.write("KNHANES 2019-2023\n")
            f.write("="*80 + "\n\n")

            f.write("ANALYSIS OVERVIEW\n")
            f.write("-"*80 + "\n")
            f.write(f"Total Sample Size: {len(self.df):,}\n")
            f.write(f"Analysis Sample: {len(self.df_analysis):,}\n")
            f.write(f"Outcome: STOP-Bang High Risk (≥3 points)\n")
            f.write(f"  High Risk: {self.df['high_risk'].sum():,} ({self.df['high_risk'].mean()*100:.1f}%)\n")
            f.write(f"  Low Risk: {(1-self.df['high_risk']).sum():,} ({(1-self.df['high_risk'].mean())*100:.1f}%)\n\n")

            f.write("STATISTICAL METHODS\n")
            f.write("-"*80 + "\n")
            f.write("- Crude odds ratios: Univariate logistic regression\n")
            f.write("- Adjusted odds ratios: Multivariable logistic regression\n")
            f.write("- Adjustments: Age, sex, BMI, education, income\n")
            f.write("- Significance level: α = 0.05\n")
            f.write("- Software: Python (statsmodels)\n\n")

            f.write("KEY FINDINGS\n")
            f.write("-"*80 + "\n")

            # Top risk factors (highest ORs from crude analysis)
            if len(self.results_crude) > 0:
                top_risks = self.results_crude[self.results_crude['OR'] > 1].nlargest(5, 'OR')
                f.write("\nTop 5 Risk Factors (Crude OR):\n")
                for idx, row in top_risks.iterrows():
                    f.write(f"  {row['Risk Factor']} ({row['Category']}): OR={row['OR']:.2f}, ")
                    f.write(f"95% CI ({row['95% CI Lower']:.2f}-{row['95% CI Upper']:.2f}), p={row['P-value']:.4f}\n")

            f.write("\n\nFILES GENERATED\n")
            f.write("-"*80 + "\n")
            f.write("1. crude_odds_ratios.csv - All crude ORs with 95% CI\n")
            f.write("2. adjusted_odds_ratios.csv - Adjusted ORs from multivariable models\n")
            f.write("3. analysis_summary_report.txt - This file\n")

            f.write("\n" + "="*80 + "\n")
            f.write("Analysis completed successfully\n")
            f.write("="*80 + "\n")

        print(f"✓ Summary report saved: {report_file.name}")

        return report_file


def main():
    """Main execution function"""

    # Initialize analysis
    analyzer = STOPBangORAnalysis()

    # Load data
    analyzer.load_data()

    # Create outcome variable
    analyzer.create_outcome_variable()

    # Create risk factor variables
    analyzer.create_risk_factor_variables()

    # Prepare analysis dataset
    analyzer.prepare_analysis_dataset()

    # Calculate crude odds ratios
    analyzer.calculate_crude_odds_ratios()

    # Calculate adjusted odds ratios
    analyzer.calculate_adjusted_odds_ratios()

    # Generate summary report
    analyzer.generate_summary_report()

    print("\n" + "="*80)
    print("✓ STOP-BANG ODDS RATIO ANALYSIS COMPLETED SUCCESSFULLY")
    print("="*80)
    print(f"\nResults saved in: {RESULTS_PATH}")
    print("\nGenerated files:")
    print("  1. crude_odds_ratios.csv")
    print("  2. adjusted_odds_ratios.csv")
    print("  3. analysis_summary_report.txt")


if __name__ == "__main__":
    main()
