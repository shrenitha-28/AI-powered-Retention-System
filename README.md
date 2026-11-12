# 🤖 AI Powered Retention System

An intelligent machine learning web application that predicts whether a customer is likely to churn by analyzing customer demographics, service subscriptions, billing information, payment behavior, and contract details.

The system automates the complete prediction pipeline—from preprocessing user inputs using the same transformations applied during model training to generating accurate churn predictions with confidence scores.

---

## 📌 Project Overview

Customer retention is one of the biggest challenges for subscription-based businesses. Acquiring a new customer costs significantly more than retaining an existing one.

The AI Powered Retention System helps businesses identify customers who are likely to leave so that proactive retention strategies can be implemented.

The application is built using:

- Python
- Flask
- Scikit-Learn
- Pandas
- NumPy
- HTML
- CSS
- JavaScript

---

## 🚀 Features

- Customer Churn Prediction
- Probability Score Generation
- Automatic Data Preprocessing
- Automatic Feature Engineering
- Consistent Model Pipeline
- Flask Web Interface
- Pickle-based Model Deployment
- Real-time Predictions

---

## 🧠 Machine Learning Workflow

### 1. Data Collection

- Telco Customer Churn Dataset

---

### 2. Data Cleaning

- Missing Value Handling
- Duplicate Removal
- Data Type Conversion
- Feature Formatting

---

### 3. Feature Engineering

Numerical Features

- Tenure
- MonthlyCharges
- TotalCharges

Categorical Features

- Gender
- SeniorCitizen
- Partner
- Dependents
- PhoneService
- MultipleLines
- InternetService
- OnlineSecurity
- OnlineBackup
- DeviceProtection
- TechSupport
- StreamingTV
- StreamingMovies
- Contract
- PaperlessBilling
- PaymentMethod
- SIM

---

### 4. Encoding

#### Label Encoding

Applied on Binary Features

- Gender
- SeniorCitizen
- Partner
- Dependents
- PhoneService
- PaperlessBilling

---

#### One Hot Encoding

Applied on

- MultipleLines
- InternetService
- OnlineSecurity
- OnlineBackup
- DeviceProtection
- TechSupport
- StreamingTV
- StreamingMovies
- PaymentMethod
- SIM

---

#### Ordinal Encoding

Applied on

Contract

Encoding Order

```
Month-to-month → 0
One year        → 1
Two year        → 2
```

---

### 5. Numerical Transformation

Applied Quantile Transformation independently on

- TotalCharges
- MonthlyCharges
- Tenure

Saved Transformers

- TotalCharges_quantile.pkl
- MonthlyCharges_quantile.pkl
- tenure_quantile.pkl

---

### 6. Outlier Handling

Used IQR-based Trimming after Quantile Transformation.

---

### 7. Data Balancing

Random Over Sampling (ROS)

Used to balance

```
Churn = Yes
Churn = No
```

---

### 8. Model Training

The following models were trained and evaluated:

- K-Nearest Neighbors
- Naive Bayes
- Logistic Regression
- Decision Tree
- Random Forest
- AdaBoost
- Gradient Boosting
- XGBoost
- Support Vector Machine

---

### 9. Final Model

Selected Model

**Logistic Regression**

Reasons

- High Recall
- Good Precision
- Balanced F1 Score
- Stable Probability Predictions
- Suitable for Customer Retention Problems

Saved Model

```
Customer.pkl
```

---

## 💾 Saved Pickle Files

```
Customer.pkl
```

Machine Learning Model

```
onehot_encoder.pkl
```

One Hot Encoder

```
ordinal_encoder.pkl
```

Ordinal Encoder

```
TotalCharges_quantile.pkl
```

Quantile Transformer

```
MonthlyCharges_quantile.pkl
```

Quantile Transformer

```
tenure_quantile.pkl
```

Quantile Transformer

---

## 🌐 Deployment Workflow

```
User Input

        │

        ▼

Label Encoding

        │

        ▼

One Hot Encoding

        │

        ▼

Ordinal Encoding

        │

        ▼

Quantile Transformation

        │

        ▼

Feature Vector

        │

        ▼

Logistic Regression Model

        │

        ▼

Prediction

        │

        ▼

Probability Score
```

---

## 📂 Project Structure

```
AI Powered Retention System
│
├── app.py
├── Customer.pkl
├── onehot_encoder.pkl
├── ordinal_encoder.pkl
├── TotalCharges_quantile.pkl
├── MonthlyCharges_quantile.pkl
├── tenure_quantile.pkl
│
├── templates
│      └── index.html
│
├── static
│      ├── style.css
│      └── script.js
│
├── train_dataset.csv
├── test_dataset.csv
│
├── requirements.txt
│
└── README.md
```

---

## ⚙️ Technologies Used

### Programming Language

- Python

### Machine Learning

- Scikit-Learn
- XGBoost

### Data Processing

- Pandas
- NumPy

### Visualization

- Matplotlib
- Seaborn

### Backend

- Flask

### Frontend

- HTML
- CSS
- JavaScript

---

## 📈 Output

The application predicts

- Customer is likely to Churn

or

- Customer is NOT likely to Churn

Along with

- Churn Probability (%)

---

## 🎯 Future Enhancements

- SHAP Explainability
- Feature Importance Visualization
- Customer Risk Dashboard
- Email Alerts
- Cloud Deployment
- Database Integration
- Live Customer Monitoring
- Batch Prediction Support

---

## 👩‍💻 Author

**Mohana Priya Korukoppula**

AI & Machine Learning Engineer

GitHub: https://github.com/MohanaPriya2107

LinkedIn: https://www.linkedin.com/in/mohanapriya-korukoppula/

---

## ⭐ If you found this project useful

Please consider giving it a ⭐ on GitHub.
