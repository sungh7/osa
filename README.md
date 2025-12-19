# Two-Stage OSA Screening Framework

This repository contains the source code, trained models, and analysis scripts for the research paper: **"A Two-Stage Screening Framework for Obstructive Sleep Apnea: Integrating Population Surveillance and HSAT-Based Triage"**.

## 📂 Repository Structure

*   `src/`: Python source code.
    *   `train_hsat_triage_model.py`: Training script for the Stage 2 Triage Model (Random Forest).
    *   `analyze_stage1_clean.py`: Epidemiology analysis script for the Stage 1 Screening.
    *   `draw_framework_diagram.py`: Visualization generator for the framework methodology.
*   `models/`: Pre-trained model artifacts.
    *   `hsat_triage_model.pkl`: The final trained Random Forest classifier (Stage 2).
    *   `stage1_screening_model.pkl`: Logistic Regression model for STOP-Bang screening probability.
    *   `neck_imputer_model.pkl`: Helper model for Stage 1 neck circumference estimation.
*   `results/`: Generated figures and metrics.
    *   `18_population_risk_heatmap.png`: Population risk stratification.
    *   `19_roc_comparison.png`: Model performance comparison (ROC).
    *   `20_confusion_matrix.png`: Clinical utility validation.
*   `data/`: Data directory.
    *   `sample/`: **Synthetic sample data** for verifying code execution.

## 🚀 Usage & Reproducibility

### 1. Installation
Install the required dependencies:
```bash
pip install -r requirements.txt
```

### 2. Data Availability
Due to privacy regulations (Korea Disease Control and Prevention Agency), the raw KNHANES and PSG datasets cannot be publicly shared.
However, we provide **synthetic sample data** in `data/sample/` that mimics the structure and statistical properties of the original dataset. This allows reviewers and researchers to execute the code and verify the analysis pipeline.

### 3. Running the Analysis (on Sample Data)
To test the Triage Model training pipeline using the sample data:
```bash
# Ensure the script points to data/sample/sample_psg.csv if real data is absent
python src/train_hsat_triage_model.py
```

### 4. Visualizing the Framework
To regenerate the methodology diagram:
```bash
python src/draw_framework_diagram.py
```

## 📊 Key Findings
*   **Stage 1 (Population):** 37.3% of the target population screens positive on STOP-Bang (Score ≥ 3).
*   **Stage 2 (Triage):** The HSAT-based Random Forest model achieves an **AUC of 0.905**, effectively filtering false positives before PSG referral.

**License:** MIT
