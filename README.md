# 🏦 Bank Customer Churn Prediction API

A machine learning project that predicts whether a bank customer is likely to leave (churn), served as a live API and containerized with Docker — built end-to-end from raw data to deployment.

---

# Live Demo -  https://bank-churn-prediction-lpda.onrender.com/docs

** Live demo through "render.com"

## 📌 Problem Statement

Banks lose money every time a customer leaves. But retention teams can't personally check on every single customer — it doesn't scale.

**This project builds a model that looks at a customer's profile and predicts the probability that they will churn (leave the bank).** This lets a business prioritize outreach — focusing retention efforts on high-risk customers instead of guessing.

- **Input:** Customer details (credit score, age, balance, activity status, etc.)
- **Output:** A churn probability (0–100%) and a risk label (`High` / `Low`)
- **Type of problem:** Binary classification (`Exited` = 1 if churned, 0 if stayed)

---

## Our journey --
✅ Data cleaning + EDA
✅ Model comparison (Logistic Regression → Random Forest)
✅ Hyperparameter tuning
✅ Caught and fixed data leakage bug
✅ Caught and fixed scaling mismatch bug
✅ FastAPI wrapper with validation
✅ Dockerized
✅ Pushed to GitHub with clean structure + README
✅ Deployed live on Render

## 📊 Dataset

- **Source:** Bank Customer Churn dataset (`Churn_Modelling.csv`)
- **Rows:** ~10,000 customers
- **Target column:** `Exited` (1 = churned, 0 = stayed)

| Column | Description |
|---|---|
| CreditScore | Customer's credit score |
| Geography | Country (France / Germany / Spain) |
| Gender | Male / Female |
| Age | Customer's age |
| Tenure | Years as a bank customer |
| Balance | Account balance |
| NumOfProducts | Number of bank products used |
| HasCrCard | Has a credit card (0/1) |
| IsActiveMember | Is an active member (0/1) |
| EstimatedSalary | Estimated salary |
| Exited | 🎯 Target — did the customer churn? |

Dropped columns: `RowNumber`, `CustomerId`, `Surname` — pure identifiers with no predictive value.

---

## 🧹 Data Preparation

1. Dropped identifier columns (no predictive signal)
2. One-hot encoded `Geography` and `Gender` (models need numbers, not text)
3. Split data into **80% train / 20% test**, using `stratify=y` to preserve the churn ratio in both sets
4. Scaled features for linear models (Random Forest doesn't need this — trees split on thresholds, not distances)

---

## 🤖 Model Evaluation Journey — The Real Story

This project wasn't a straight line to a good model — here's what was actually tried, what went wrong, and how it was fixed. This is arguably the most valuable part of the project.

### 1️⃣ Baseline: Logistic Regression
- **Result:** 80.9% accuracy, but only **20.6% recall** 😬
- **Problem discovered:** High accuracy was misleading. Since only ~20% of customers churn, the model was mostly just predicting "no churn" for everyone and still looking "accurate."

### 2️⃣ Logistic Regression + Class Balancing
- Applied `class_weight='balanced'` to force the model to pay attention to the minority (churn) class
- **Result:** Recall jumped to 69.5%, but precision dropped to 37.9% — more false alarms, but far fewer missed churners
- **Insight:** For churn prediction, missing a real churner (false negative) is usually costlier than a false alarm — so this tradeoff was worth it

### 3️⃣ Random Forest (Untuned)
- Switched to a tree-based model to capture non-linear patterns and feature interactions that Logistic Regression can't
- **Result:** ROC-AUC improved to 73.9% — the best yet

### 4️⃣ 🚨 The "Perfect Model" Bug
- An early Random Forest run scored **100% accuracy, 100% recall, 100% everything**
- 🔍 **This was a huge red flag, not a win.** A perfect score on real-world churn data essentially never happens.
- **Root cause found:** `rf.fit(X_test, y_test)` — the model was accidentally **trained on the test set**, then evaluated on that same data. It hadn't learned anything general; it had just memorized the answers, like a student grading their own answer key.
- **Fix:** Corrected to `rf.fit(X_train, y_train)`, then evaluated on the untouched `X_test`.
- **Lesson:** Suspiciously perfect metrics are a sign to investigate the pipeline, not celebrate.

### 5️⃣ Hyperparameter Tuning (RandomizedSearchCV)
- Searched across `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, and `max_features` using 5-fold cross-validation, optimizing for ROC-AUC
- **Result:** ROC-AUC improved to **76.3%**, with recall rising to 68% — the best balance found

### 6️⃣ 🚨 The Scaling Mismatch Bug (found via the API)
- Once deployed behind the API, predictions came back nonsensical — an obviously loyal customer was flagged "High risk," and an obviously risky customer was flagged "Low risk"
- 🔍 **Root cause found:** The model had been trained on *scaled* data (`X_train_scaled`), but the live API was sending *raw, unscaled* customer data — a mismatch between training and real-world input
- **Fix:** Retrained Random Forest on the original unscaled `X_train` (which trees don't need anyway), then re-saved the model
- **Lesson:** Always keep training and inference (prediction) data in **exactly the same format** — this is one of the most common real-world ML bugs

### ✅ Final Model: Tuned Random Forest (trained on unscaled data)

| Metric | Score |
|---|---|
| Accuracy | 81.3% |
| Precision | 53.2% |
| Recall | 68.1% |
| F1 Score | 59.7% |
| **ROC-AUC** | **76.4%** |

---

## 💡 Key Business Insight

Feature importance analysis revealed what actually drives churn in this data:

| Feature | Importance |
|---|---|
| **Age** | 35.9% 🥇 |
| **NumOfProducts** | 21.5% 🥈 |
| Balance | 11.0% |
| EstimatedSalary | 6.8% |
| CreditScore | 6.4% |
| IsActiveMember | 5.7% |

🎯 **Churn rate by age group** — a surprising non-linear pattern:

| Age Group | Churn Rate |
|---|---|
| 18–30 | 7.5% |
| 30–40 | 12.1% |
| 40–50 | 34.0% |
| **50–60** | **56.2%** 🔥 (peak risk) |
| 60+ | 24.8% |

Churn rises sharply through middle age, **peaks at 50–60**, then drops again — a pattern a simple linear model would likely miss, but Random Forest naturally captures.

Customers with **3–4 products churn more**, likely a sign of over-selling or dissatisfaction, even among otherwise "active" customers.

---

## 🚀 Tech Stack

| Purpose | Tool |
|---|---|
| Modeling | scikit-learn (Random Forest) |
| API | FastAPI |
| Server | Uvicorn |
| Model persistence | joblib |
| Containerization | Docker |
| Version control | Git + GitHub |

---

## 📁 Project Structure

```
bank_churn_prediction/
├── api/
│   └── main.py              # FastAPI app
├── models/
│   ├── bank_churn_model.pkl # Trained model
│   └── model_columns.pkl    # Column order the model expects
├── Dockerfile
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🔧 How Everything Works — Key Concepts

### 📦 What is `joblib`?
`joblib` saves a trained Python model to a file (`.pkl`) so it doesn't disappear when you close your notebook — and loads it back later without retraining.
```python
import joblib
joblib.dump(model, "models/bank_churn_model.pkl")   # save
model = joblib.load("models/bank_churn_model.pkl")   # load
```

### 🌐 What is FastAPI / Uvicorn?
FastAPI builds the "door" (API) that lets other apps send customer data and get predictions back. Uvicorn is the engine that actually runs the FastAPI app as a live server.

### 🐳 What is Docker?
Docker packages the app + all its dependencies into a portable "box" (image) that runs identically on any machine — solving the classic "works on my machine" problem.

---

## ▶️ How to Run This Project

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/bank-churn-prediction.git
cd bank-churn-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the API locally (without Docker)
```bash
uvicorn api.main:app --reload
```
Then open: `http://127.0.0.1:8000/docs`

### 4. Run with Docker instead
```bash
docker build -t churn-api .
docker run -p 8000:8000 churn-api
```
Then open: `http://127.0.0.1:8000/docs`

---

## 📝 Example Request

**POST** `/predict`
```json
{
  "CreditScore": 650,
  "Geography": "Germany",
  "Gender": "Male",
  "Age": 40,
  "Tenure": 3,
  "Balance": 60000,
  "NumOfProducts": 2,
  "HasCrCard": 1,
  "IsActiveMember": 1,
  "EstimatedSalary": 50000
}
```

**Response**
```json
{
  "churn_probability": 0.62,
  "churn_prediction": 1,
  "risk_level": "High"
}
```

---

## 🛠️ Git Commands Used

```bash
git init                                   # start tracking this project
git add .                                  # stage all files
git commit -m "message"                    # save a snapshot
git branch -M main                         # rename branch to main
git remote add origin <repo-url>           # link to GitHub
git push -u origin main                    # upload to GitHub
git rm -r --cached <folder>                # stop tracking a folder (keeps it locally)
```

---

## 🔮 Future Improvements

- 🔁 Try XGBoost / LightGBM for comparison against Random Forest
- 🎯 Feature engineer an explicit `AgeGroup` bucket to help the model capture the non-linear age-churn relationship even more directly
- 📊 Add MLflow for experiment tracking
- ☁️ Deploy live on Render with CI/CD via GitHub Actions
- 🧪 Add automated tests (`pytest`) for the API

---

## ⚠️ Honest Limitations

- ROC-AUC of 76% is decent but not exceptional — there's real room for improvement
- Precision (53%) means roughly half of "High risk" flags are false alarms — acceptable for this use case since missing a churner is costlier, but worth knowing
- Trained on a single static dataset — a production system would need periodic retraining as customer behavior shifts over time
