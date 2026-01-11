import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn
import pickle
import os

# Set MLflow tracking
mlflow.set_experiment("churn-prediction")

def load_and_preprocess_data(filepath):
    """Load and preprocess the churn dataset"""
    df = pd.read_csv(filepath)
    
    # Drop unnecessary columns
    df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)
    
    # Encode categorical variables
    le_geography = LabelEncoder()
    le_gender = LabelEncoder()
    
    df['Geography'] = le_geography.fit_transform(df['Geography'])
    df['Gender'] = le_gender.fit_transform(df['Gender'])
    
    # Split features and target
    X = df.drop('Exited', axis=1)
    y = df['Exited']
    
    return X, y, le_geography, le_gender

def train_model(X_train, y_train, params):
    """Train Random Forest model"""
    model = RandomForestClassifier(
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        min_samples_split=params['min_samples_split'],
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate model and return metrics"""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    return metrics

def main():
    print("Loading data...")
    X, y, le_geography, le_gender = load_and_preprocess_data('data/Churn_Modelling.csv')
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Hyperparameters to try
    params_list = [
        {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 5},
        {'n_estimators': 200, 'max_depth': 15, 'min_samples_split': 2},
        {'n_estimators': 150, 'max_depth': 20, 'min_samples_split': 3},
    ]
    
    best_model = None
    best_score = 0
    
    # Train multiple models with MLflow tracking
    for i, params in enumerate(params_list):
        with mlflow.start_run(run_name=f"rf_model_{i+1}"):
            print(f"\nTraining model {i+1} with params: {params}")
            
            # Log parameters
            mlflow.log_params(params)
            
            # Train model
            model = train_model(X_train_scaled, y_train, params)
            
            # Evaluate
            metrics = evaluate_model(model, X_test_scaled, y_test)
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            print(f"Results: Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1_score']:.4f}, ROC-AUC={metrics['roc_auc']:.4f}")
            
            # Track best model
            if metrics['roc_auc'] > best_score:
                best_score = metrics['roc_auc']
                best_model = model
            
            # Log model
            mlflow.sklearn.log_model(model, "model")
    
    # Save best model and preprocessing objects
    print(f"\nBest model ROC-AUC: {best_score:.4f}")
    
    os.makedirs('models', exist_ok=True)
    
    with open('models/model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    with open('models/le_geography.pkl', 'wb') as f:
        pickle.dump(le_geography, f)
    
    with open('models/le_gender.pkl', 'wb') as f:
        pickle.dump(le_gender, f)
    
    print("\nModel and preprocessing objects saved!")
    print("Feature names:", list(X.columns))

if __name__ == "__main__":
    main()