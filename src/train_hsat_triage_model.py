import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve, precision_recall_curve
from pathlib import Path

# Paths
BASE_PATH = Path('/data/knhanes/paper_revision')
DATA_PATH = BASE_PATH / 'data'
RESULTS_PATH = BASE_PATH / 'results'
MODELS_PATH = BASE_PATH / 'models'

# Create dirs
RESULTS_PATH.mkdir(exist_ok=True)
MODELS_PATH.mkdir(exist_ok=True)

def load_data():
    print("Loading data from new revision folder...")
    df_osa = pd.read_csv(DATA_PATH / 'OSA_enhanced.csv')
    df_non_osa = pd.read_csv(DATA_PATH / 'non_OSA_enhanced.csv')
    
    # Combine
    df = pd.concat([df_osa, df_non_osa], ignore_index=True)
    return df

def preprocess_data(df):
    print("Preprocessing data for HSAT-based Triage...")
    
    # 1. Define Target: Severe OSA (AHI >= 30) -> High Priority Referral
    # Note: Using AHI only to create labels. NOT as feature.
    df['AHI'] = pd.to_numeric(df['AHI'], errors='coerce').fillna(0)
    df['Target_Severe'] = (df['AHI'] >= 30).astype(int)
    
    # 2. Define Features (HSAT Compatible)
    # Demographics
    df['Is_Male'] = df['Sex'].apply(lambda x: 1 if str(x).upper() in ['M', 'MALE', '1', '1.0'] else 0)
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
    df['BMI'] = pd.to_numeric(df['BMI'], errors='coerce')
    
    # Neck (Imputed or Measured - assuming 'Neck_Imputed' available or fallback)
    # If missing, we simulate it or drop. STOP-Bang usually needs it.
    if 'Neck_Imputed' in df.columns:
        df['Neck'] = df['Neck_Imputed']
    elif 'Neck' in df.columns:
        df['Neck'] = pd.to_numeric(df['Neck'], errors='coerce')
    else:
        # Fallback: Estimate from BMI/Sex if needed (simplified)
        print("Warning: Neck column missing, using BMI proxy if needed")
        df['Neck'] = 0 # Placeholder if absolutely missing, but RF handles it if we drop
        
    # HSAT Features (PulseOx)
    # ODI: Oxygen Desaturation Index
    df['ODI'] = pd.to_numeric(df['ODI'], errors='coerce').fillna(0)
    
    # SpO2 Min
    if 'O2_min' in df.columns:
        df['Min_SpO2'] = pd.to_numeric(df['O2_min'], errors='coerce').fillna(95)
    else:
        df['Min_SpO2'] = 95
        
    # SpO2 Mean
    if 'O2_mean' in df.columns:
        df['Mean_SpO2'] = pd.to_numeric(df['O2_mean'], errors='coerce').fillna(98)
    else:
        df['Mean_SpO2'] = 98
        
    # Time < 90%
    if 'perc_O2_\nunder_90' in df.columns: # Check column name formatting
        df['Time_Under_90'] = pd.to_numeric(df['perc_O2_\nunder_90'], errors='coerce').fillna(0)
    elif 'perc_O2_under_90' in df.columns:
        df['Time_Under_90'] = pd.to_numeric(df['perc_O2_under_90'], errors='coerce').fillna(0)
    else:
        df['Time_Under_90'] = 0

    # 3. Select Final Feature Vectors
    features = ['Age', 'Is_Male', 'BMI', 'Neck', 'ODI', 'Min_SpO2', 'Mean_SpO2', 'Time_Under_90']
    
    print(f"Selected Features: {features}")
    
    # Drop NaNs in features
    df_clean = df.dropna(subset=features)
    
    X = df_clean[features]
    y = df_clean['Target_Severe']
    
    print(f"Data shape: {X.shape}")
    print("Class Distribution (Severe OSA):")
    print(y.value_counts())
    
    return X, y, features

def train_and_evaluate(X, y, feature_names):
    print("\nTraining Triage Model (Random Forest)...")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # Train
    clf = RandomForestClassifier(n_estimators=300, max_depth=10, class_weight='balanced', random_state=42)
    clf.fit(X_train, y_train)
    
    # Predict
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    
    # Metrics
    print("\nModel Evaluation (Test Set):")
    print(classification_report(y_test, y_pred))
    
    auc = roc_auc_score(y_test, y_prob)
    print(f"AUC: {auc:.4f}")
    
    # Specificity/Sensitivity
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    ppv = tp / (tp + fp)
    npv = tn / (tn + fn)
    
    print(f"Sensitivity: {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"PPV (Clinical Prevalence): {ppv:.4f}")
    print(f"NPV: {npv:.4f}")
    
    # Feature Importance
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print("\nFeature Ranking:")
    for f in range(X.shape[1]):
        print(f"{f+1}. {feature_names[indices[f]]}: {importances[indices[f]]:.4f}")
        
    # Save Results
    results = {
        'AUC': auc,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'PPV': ppv,
        'Feature_Imp': dict(zip(feature_names, importances))
    }
    
    # Plot ROC
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=f'HSAT Triage Model (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve: Stage 2 HSAT Triage')
    plt.legend()
    plt.savefig(RESULTS_PATH / 'triage_roc_curve.png')
    plt.close()
    
    return clf, results

def main():
    df = load_data()
    X, y, feature_names = preprocess_data(df)
    model, results = train_and_evaluate(X, y, feature_names)
    
    # Save model
    joblib.dump(model, MODELS_PATH / 'hsat_triage_model.pkl')
    
    # Save metrics txt
    with open(RESULTS_PATH / 'model_metrics.txt', 'w') as f:
        f.write(str(results))

if __name__ == "__main__":
    main()
