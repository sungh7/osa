# A Two-Stage Screening Framework for Obstructive Sleep Apnea: Integrating Population Surveillance and HSAT-Based Triage

**Abstract**

**Background:** Obstructive Sleep Apnea (OSA) is highly prevalent but underdiagnosed. Standard screening tools like STOP-Bang have high sensitivity but low specificity, leading to excessive referrals for expensive Polysomnography (PSG).
**Objective:** To estimate the burden of OSA screening positivity in the Korean population and develop a secondary triage model using Home Sleep Apnea Test (HSAT)-compatible features to prioritize high-risk patients.
**Methods:** We analyzed data from the Korea National Health and Nutrition Examination Survey (KNHANES 2019-2023, N=20,906) to determine the prevalence of STOP-Bang high-risk (score ≥3) individuals. Subsequently, using a clinical PSG database (N=779), we trained a Random Forest triage model to predict Severe OSA (AHI ≥30) using only demographics and HSAT-derived variables (ODI, SpO2), excluding diagnostic indices (AHI, RDI).
**Results:** The population prevalence of STOP-Bang high-risk status was 37.3% in adults ≥40 years, with a marked sex disparity (Men 78.1% vs. Women 6.1%) driven by scoring structure. In the clinical validation, the HSAT-based triage model achieved an AUC of **0.905**, Sensitivity of 86.1%, and Specificity of 78.6%, significantly outperforming the baseline STOP-Bang specificity (<40%).
**Conclusion:** A massive burden of screening-positive individuals exists in the general population. A two-stage strategy incorporating a secondary HSAT-based triage model can effective filter false positives, optimizing resource allocation for confirmatory PSG.

---

## 1. Introduction

Obstructive Sleep Apnea (OSA) is a major public health concern associated with cardiovascular morbidity, metabolic dysregulation, and impaired quality of life. Despite its high prevalence, estimated at 15-30% in males and 10-15% in females, the majority of cases remain undiagnosed. The gold standard for diagnosis, Polysomnography (PSG), is resource-intensive, expensive, and limited by facility availability.

To address this, screening questionnaires like STOP-Bang have been widely adopted. While STOP-Bang demonstrates excellent sensitivity (90-100% for severe OSA), its specificity is notoriously low (often <40%), resulting in a high rate of false-positive referrals that overwhelm sleep clinics. This necessitates a **Two-Stage Screening Strategy**, where a broad primary screen is followed by a more specific secondary triage before PSG referral.

In this study, we propose an integrated framework:
1.  **Stage 1 Assessment:** Quantifying the "Screening Burden" in the general population using KNHANES data (N=20,906).
2.  **Stage 2 Triage:** Developing a Machine Learning triage model that utilizes only variables accessible via Home Sleep Apnea Testing (HSAT)—specifically Oxygen Desaturation Index (ODI) and SpO2 metrics—to identify patients requiring priority in-lab PSG.

## 2. Methods

### 2.1 Data Sources
*   **Population Dataset (Stage 1):** KNHANES VIII (2019-2023), a nationally representative cross-sectional survey. We analyzed 20,906 adults. Neck circumference (NC), missing in standard surveys, was imputed using a validated regression model ($R^2=0.86$).
*   **Clinical Dataset (Stage 2):** A retrospective cohort of 779 patients undergoing PSG at a specialized sleep center.

### 2.2 Stage 1: Population Epidemiology
We determined the prevalence of "High Risk" status (STOP-Bang ≥ 3). We assessed associations with 43 lifestyle and clinical factors. 
*   **Methodological Note:** We prioritized **Crude Odds Ratios (ORs)** for descriptive analysis. We deliberately avoided multivariable adjustment for STOP-Bang components (Age, Sex, BMI) to prevent **Outcome Variable Circularity**, a structural bias where adjusting for scoring criteria creates artificial "protective" associations.

### 2.3 Stage 2: HSAT-Based Triage Model
We developed a Random Forest Classifier to predict **Severe OSA (AHI ≥ 30)**.
*   **Inclusion Criteria:** Patients with complete Demographic and Oximetry data.
*   **Feature Selection (Strict Triage Horizon):** To prevent data leakage, we excluded all PSG-derived diagnostic indices (AHI, RDI, Arousal Index). The model included only:
    *   *Demographics:* Age, Sex, BMI, Neck Circumference.
    *   *HSAT-Compatible Metrics:* Oxygen Desaturation Index (ODI), Mean SpO2, Minimum SpO2, % Time SpO2 < 90%.
*   **Validation:** Stratified Train/Test split (70/30) with class balancing. Evaluation metrics included Area Under the Curve (AUC), Sensitivity, Specificity, and Positive Predictive Value (PPV).

## 3. Results

### 3.1 Stage 1: The Burden of Screening Positivity
In the Korean adult population (≥40 years), **37.3%** were classified as High Risk (STOP-Bang ≥ 3). 
*   **Sex Disparity:** 78.1% of Men vs. 6.1% of Women screened positive. This 12-fold difference reflects the structural scoring of STOP-Bang (Male sex = 1 point), meaning men ≥50 years (2 points total) require only one symptom to cross the high-risk threshold.
*   **Crude Associations:** Strong observational associations were found for Diabetes (OR 5.62), Hypertension (OR 4.23), and High Sodium Intake (OR 3.61).

### 3.2 Stage 2: Triage Model Performance
The Random Forest model utilizing HSAT-compatible features achieved robust discrimination for Severe OSA.

| Metric | Value | 95% CI |
| :--- | :--- | :--- |
| **AUC** | **0.9052** | 0.88 - 0.93 |
| **Sensitivity** | 86.1% | 81.0 - 90.5% |
| **Specificity** | 78.6% | 71.2 - 85.1% |
| **PPV** (Clinical Preval.) | 77.5% | - |

*   **Feature Importance:** The top predictors were **% Time SpO2 < 90%** and **Mean SpO2**, confirming that oximetry kinetics are the most critical signals for severity prediction, outperforming simple demographics.

## 4. Discussion

### 4.1 From Screening to Triage
Our findings highlight the specific limitation of STOP-Bang: a 37% "High Risk" prevalence in the general population is too broad to serve as a direct referral criterion, given the true disease prevalence is estimated at 15-20%. The massive volume of false positives necessitates a filter. 
Our Stage 2 model demonstrates that adding simple oximetry data (HSAT) transforms the screening process. With an AUC of 0.905, this "Enhanced Triage" strategy can filter out ~78% of non-severe cases (Specificity) while retaining ~86% of severe cases for priority care.

### 4.2 Handling Methodological Artifacts
A key insight from this study is the identification of **Outcome Circularity**. Previous attempts to "adjust" risk factor analyses for Age/Sex/BMI resulted in paradoxical protective effects for obesity. We advocate for interpreting STOP-Bang epidemiologic associations carefully, recognizing them as "Risk Markers for Screening Positivity" rather than independent causal factors for OSA.

### 4.3 Limitations
*   **Stage 1:** Reliance on self-reported symptoms in KNHANES.
*   **Stage 2:** The HSAT metrics (ODI, SpO2) were derived from PSG channels. While theoretically identical to independent HSAT devices, future validation using actual wearable data is recommended.

## 5. Conclusion
Integrating broad population screening with an HSAT-based triage model offers a pragmatic solution to the OSA diagnostic bottleneck. By leveraging oximetry data in a second stage, healthcare systems can drastically reduce unnecessary PSG referrals without compromising the detection of severe cases.

**Conflict of Interest:** None declared.
**Funding:** None.
