# churn_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, classification_report,
                             confusion_matrix, roc_curve)
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

# Load data
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

# Data preprocessing
def preprocess_data(df):
    df_processed = df.copy()
    
    # Remove non-informative features
    df_processed = df_processed.drop(['CustomerId', 'Surname'], axis=1)
    
    # Handle missing values (if any)
    df_processed = df_processed.dropna()
    
    # Encode categorical variables
    df_processed = pd.get_dummies(df_processed, columns=['Geography', 'Gender'], drop_first=True)
    
    return df_processed

# Feature engineering
def engineer_features(df):
    df_eng = df.copy()
    
    # Balance-to-Salary ratio
    df_eng['Balance_Salary_Ratio'] = df_eng['Balance'] / (df_eng['EstimatedSalary'] + 1)
    
    # Product density: number of products per tenure
    df_eng['Product_Density'] = df_eng['NumOfProducts'] / (df_eng['Tenure'] + 1)
    
    # Age-Tenure interaction
    df_eng['Age_Tenure_Ratio'] = df_eng['Age'] / (df_eng['Tenure'] + 1)
    
    # Engagement-Product interaction
    df_eng['Engagement_Product'] = df_eng['IsActiveMember'] * df_eng['NumOfProducts']
    
    # Credit score categories
    df_eng['Credit_Score_Good'] = (df_eng['CreditScore'] >= 700).astype(int)
    df_eng['Credit_Score_Fair'] = ((df_eng['CreditScore'] >= 600) & (df_eng['CreditScore'] < 700)).astype(int)
    df_eng['Credit_Score_Poor'] = (df_eng['CreditScore'] < 600).astype(int)
    
    # Has high balance
    df_eng['High_Balance'] = (df_eng['Balance'] > df_eng['Balance'].median()).astype(int)
    
    # Active with credit card
    df_eng['Active_With_Card'] = (df_eng['IsActiveMember'] & df_eng['HasCrCard']).astype(int)
    
    # Age groups
    df_eng['Age_Young'] = (df_eng['Age'] < 30).astype(int)
    df_eng['Age_Middle'] = ((df_eng['Age'] >= 30) & (df_eng['Age'] < 50)).astype(int)
    df_eng['Age_Senior'] = (df_eng['Age'] >= 50).astype(int)
    
    # Tenure groups
    df_eng['Tenure_Short'] = (df_eng['Tenure'] < 3).astype(int)
    df_eng['Tenure_Medium'] = ((df_eng['Tenure'] >= 3) & (df_eng['Tenure'] < 7)).astype(int)
    df_eng['Tenure_Long'] = (df_eng['Tenure'] >= 7).astype(int)
    
    return df_eng

# Main pipeline
def run_pipeline(file_path):
    # Load data
    print("Loading data...")
    df = load_data(file_path)
    print(f"Dataset shape: {df.shape}")
    
    # Check target distribution
    print(f"Churn distribution:\n{df['Exited'].value_counts(normalize=True)}")
    
    # Preprocess
    print("\nPreprocessing data...")
    df_processed = preprocess_data(df)
    
    # Feature engineering
    print("Engineering features...")
    df_engineered = engineer_features(df_processed)
    
    # Separate features and target
    X = df_engineered.drop('Exited', axis=1)
    y = df_engineered['Exited']
    
    print(f"Features shape: {X.shape}")
    print(f"Total features: {len(X.columns)}")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Feature names
    feature_names = X.columns.tolist()
    
    # Train models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'XGBoost': xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    }
    
    results = {}
    trained_models = {}
    
    print("\n" + "="*60)
    print("MODEL TRAINING AND EVALUATION")
    print("="*60)
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Handle XGBoost separately (it doesn't need scaled data)
        if name == 'XGBoost':
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        
        # Cross-validation
        if name == 'XGBoost':
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
        else:
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
        
        results[name] = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'ROC-AUC': roc_auc,
            'CV_Mean': cv_scores.mean(),
            'CV_Std': cv_scores.std()
        }
        
        trained_models[name] = model
        
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  ROC-AUC: {roc_auc:.4f}")
        print(f"  CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    # Feature importance for Random Forest
    rf_model = trained_models['Random Forest']
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\n" + "="*60)
    print("TOP 10 FEATURE IMPORTANCES (Random Forest)")
    print("="*60)
    print(feature_importance.head(10).to_string(index=False))
    
    # Save models and artifacts
    print("\nSaving models and artifacts...")
    joblib.dump(trained_models, 'models/trained_models.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(feature_names, 'models/feature_names.pkl')
    joblib.dump(results, 'models/results.pkl')
    feature_importance.to_csv('models/feature_importance.csv', index=False)
    
    print("\n" + "="*60)
    print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nFiles saved in 'models/' folder:")
    print("  - trained_models.pkl")
    print("  - scaler.pkl")
    print("  - feature_names.pkl")
    print("  - results.pkl")
    print("  - feature_importance.csv")
    
    return trained_models, scaler, feature_names, results, feature_importance, X_train, X_test, y_train, y_test

if __name__ == "__main__":
    run_pipeline('European_Bank.csv')