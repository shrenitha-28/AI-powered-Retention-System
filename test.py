import pickle
import numpy as np

# Load model
with open("Customer.pkl", "rb") as f:
    model = pickle.load(f)

features = np.array([[
    50,     # TotalCharges (very low)
    1,      # Tenure (new customer)
    120,    # MonthlyCharges (very high)

    1,      # SeniorCitizen
    0,      # Partner
    0,      # Dependents
    1,      # PaperlessBilling

    1,      # MultipleLines_Yes

    1,      # InternetService_Fiber_optic
    0,      # InternetService_No

    0,      # OnlineSecurity_No_internet_service
    0,      # OnlineSecurity_Yes

    0,      # OnlineBackup_No_internet_service
    0,      # OnlineBackup_Yes

    0,      # DeviceProtection_No_internet_service
    0,      # DeviceProtection_Yes

    0,      # TechSupport_No_internet_service
    0,      # TechSupport_Yes

    0,      # StreamingTV_No_internet_service
    0,      # StreamingTV_Yes

    0,      # StreamingMovies_No_internet_service
    0,      # StreamingMovies_Yes

    0,      # PaymentMethod_Credit_card
    1,      # PaymentMethod_Electronic_check
    0,      # PaymentMethod_Mailed_check

    0,      # SIM_BSNL
    0,      # SIM_Jio
    1,      # SIM_Vodafone

    0       # Contract_res (Month-to-month)
]])

# Prediction
prediction = model.predict(features)[0]

# Probability
probability = model.predict_proba(features)[0]

print("="*50)

if prediction == 1:
    print("Prediction : Customer is likely to Churn")
else:
    print("Prediction : Customer is NOT likely to Churn")

print("Probability of Not Churn :", round(probability[0]*100,2), "%")
print("Probability of Churn     :", round(probability[1]*100,2), "%")

print("="*50)