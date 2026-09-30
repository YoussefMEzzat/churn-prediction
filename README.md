# 📉 Customer Churn Prediction

An end-to-end machine learning project that predicts which telecom customers are likely to leave, so the business can target them with retention offers before they churn.

## 🎯 Problem
Acquiring a new customer costs far more than keeping an existing one. The goal is to identify at-risk customers early.

- **Type:** Binary classification (`Churn`: Yes / No)
- **Challenge:** Imbalanced data (26.5% churn)
- **Primary metric:** **Recall**, since missing a customer who churns (False Negative) costs more than offering a discount to one who wouldn't (False Positive). Also tracked F1 and ROC-AUC.

## 📊 Dataset
[IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn): 7,043 customers, 21 features (demographics, services, account info).

## 🔍 Key Insights (EDA)
- **Contract:** Month-to-month customers churn ~43% vs ~3% for two-year contracts.
- **Tenure:** Churn drops from ~47% in the first year to ~10% after 4 years.
- **Internet:** Fiber optic customers churn the most (~42%).
- **Payment:** Electronic check users churn the most (~45%).
- **Add-ons:** Customers without Online Security or Tech Support churn much more.

## 🛠️ Approach
1. **Data cleaning:** `TotalCharges` had 11 blank values for new customers (`tenure = 0`), filled with 0.
2. **Train/test split** (80/20, stratified) **before** preprocessing to prevent data leakage.
3. **Preprocessing:** One-Hot Encoding for categorical features, StandardScaler for numerical, wrapped in a `ColumnTransformer` + `Pipeline`.
4. **Imbalance handling:** `class_weight='balanced'` / `scale_pos_weight`.
5. **Modeling:** Compared a dummy baseline, Logistic Regression, Random Forest, and XGBoost.
6. **Tuning:** GridSearchCV with 5-fold stratified cross-validation on Random Forest (reduced overfitting: train/test F1 gap from 0.40 to 0.06).

## 📈 Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Baseline (always "No") | 0.735 | 0.000 | 0.000 | 0.000 | 0.500 |
| **Logistic Regression** ✅ | 0.738 | 0.504 | **0.783** | 0.614 | **0.841** |
| Random Forest | 0.768 | 0.555 | 0.639 | 0.594 | 0.820 |
| Random Forest (tuned) | — | — | — | 0.633 | — |
| XGBoost | 0.770 | 0.556 | 0.668 | 0.607 | 0.820 |

**Final model: Logistic Regression.** It achieved the highest Recall and ROC-AUC, and it's interpretable, so the business can see *why* a customer is at risk. Tuned Random Forest had slightly higher F1, but not enough to justify the added complexity.

## 💡 Business Recommendations
- Focus retention on **new customers** during their first year.
- Incentivize **month-to-month** customers to switch to annual contracts.
- Review **Fiber optic** pricing and service quality.
- Promote **Tech Support** and **Online Security** add-ons.

## 📁 Project Structure
```
churn-prediction/
├── data/                  # dataset (download from Kaggle)
├── models/                # saved pipeline
├── notebooks/
│   └── 01_eda.ipynb       # EDA, preprocessing, modeling experiments
├── src/
│   └── train.py           # training script
├── app.py                 # Streamlit app
└── requirements.txt
```

## 🚀 How to Run
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt

# Download the CSV from Kaggle into data/, then:
python src/train.py
streamlit run app.py
```