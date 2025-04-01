# 🚢 Titanic Survival Prediction (Machine Learning Portfolio Project)

This project presents a full end-to-end machine learning pipeline for predicting passenger survival on the Titanic using Python, Scikit-Learn, and Pandas.

---

## 📊 Dataset
The [Titanic dataset](https://www.kaggle.com/competitions/titanic) is a classic binary classification problem used in Kaggle competitions.

- `891` rows × `12` features
- Includes passenger age, class, gender, fare, family aboard, etc.

---

## 🎯 Problem Statement
> Can we predict if a passenger survived the Titanic disaster based on available features?

---

## 🧠 Skills Demonstrated
- Exploratory Data Analysis (EDA)
- Data Cleaning (missing values, encoding, imputation)
- Feature Engineering (`AgeGroup`, one-hot encoding)
- Model Training (Random Forest, Logistic Regression)
- Model Evaluation (Accuracy, Precision, Recall, F1, Confusion Matrix)
- Saving & Reloading Models
- Predicting on Custom User Input

---

## ⚙️ Tools Used

| Tool | Purpose |
|------|---------|
| `pandas` | Data processing |
| `matplotlib`, `seaborn` | Visualization |
| `scikit-learn` | Modeling, scaling, evaluation |
| `joblib` | Save & reload models |

---

## 📈 Final Results (Random Forest)

| Metric | Score |
|--------|-------|
| Accuracy | ~85% |
| Precision | 85% |
| Recall | 77% |
| F1 Score | 80% |

✅ Logistic Regression also included for comparison.

---

## 🧪 Run it Yourself

```bash
# Requirements
pip install pandas numpy scikit-learn seaborn matplotlib joblib

# Run the notebook
jupyter notebook Titanic_Survival_Prediction_Portfolio.ipynb
