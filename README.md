# Two-Stage OSA Screening Framework

This repository contains the source code and analysis scripts for the research paper: **"A Two-Stage Screening Framework for Obstructive Sleep Apnea: Integrating Population Surveillance and HSAT-Based Triage"**.

## 📂 Repository Structure

*   `src/`: Python source code.
    *   `train_hsat_triage_model.py`: Trains the Stage 2 Triage Model (Random Forest) using HSAT features (ODI, SpO2).
    *   `analyze_stage1_clean.py`: Performs population-level STOP-Bang analysis (KNHANES).
    *   `draw_framework_diagram.py`: Generates the methodology flowchart.
*   `docs/`: Manuscript drafts.
    *   `RESEARCH_PAPER_PUBLICATION.docx`: Final submission manuscript.
*   `results/`: Generated figures and model metrics.
    *   `16_two_stage_framework_diagram.png`: Framework overview.
    *   `triage_roc_curve.png`: Model performance ROC.
*   `data/`: Data placeholders.
    *   *Note: Raw KNHANES (SAS) and PSG files are not included due to privacy regulations.*

## 🚀 Usage

### 1. Requirements
Install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Training the Triage Model
Run the `train_hsat_triage_model.py` script. Ensure your local `data/` folder contains the processed CSVs (`OSA_enhanced.csv`).
```bash
python src/train_hsat_triage_model.py
```

### 3. Generating Diagrams
```bash
python src/draw_framework_diagram.py
```

## 📊 Key Findings
*   **Stage 1 (Population):** 37.3% of the target population screens positive on STOP-Bang (Score ≥ 3).
*   **Stage 2 (Triage):** The HSAT-based Random Forest model achieves an **AUC of 0.905**, effectively filtering false positives before PSG referral.

## ⚠️ Data Privacy
This repository does not contain patient health information (PHI). Users must obtain their own access to KNHANES data (Korea Disease Control and Prevention Agency) to replicate the Stage 1 analysis.

**Author:** OSA Research Team
**License:** MIT
