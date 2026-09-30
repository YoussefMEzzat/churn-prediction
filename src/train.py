import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

DATA_PATH = Path("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
MODEL_PATH = Path("models/churn_model.joblib")

NUM_COLS = ['tenure', 'MonthlyCharges', 'TotalCharges']
CAT_COLS = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
            'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
            'PaperlessBilling', 'PaymentMethod']


def load_data(path):
    df = pd.read_csv(path)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    df = df.drop('customerID', axis=1)
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    return df


def build_pipeline():
    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), NUM_COLS),
        ('cat', OneHotEncoder(drop='if_binary', handle_unknown='ignore'), CAT_COLS)
    ], remainder='passthrough')

    model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')

    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])


def main():
    df = load_data(DATA_PATH)
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    print(classification_report(y_test, y_pred, target_names=['Stay', 'Churn']))
    print("ROC-AUC:", round(roc_auc_score(y_test, y_proba), 3))

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()