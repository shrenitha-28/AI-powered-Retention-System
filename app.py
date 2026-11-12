from flask import Flask, render_template, request
import pickle
import numpy as np
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

# ==========================
# Load Trained Model
# ==========================
with open("Customer.pkl", "rb") as f:
    model = pickle.load(f)


@app.route("/", methods=["GET", "POST"])
def index():

    prediction = None
    probability = None

    if request.method == "POST":

        try:

            # ==========================
            # Read Form Inputs
            # ==========================

            features = [

                float(request.form["TotalCharges"]),
                float(request.form["Tenure"]),
                float(request.form["MonthlyCharges"]),

                float(request.form["SeniorCitizen"]),
                float(request.form["Partner"]),
                float(request.form["Dependents"]),
                float(request.form["PaperlessBilling"]),

                float(request.form["MultipleLines_Yes"]),

                float(request.form["InternetService_Fiber_optic"]),
                float(request.form["InternetService_No"]),

                float(request.form["OnlineSecurity_No_internet_service"]),
                float(request.form["OnlineSecurity_Yes"]),

                float(request.form["OnlineBackup_No_internet_service"]),
                float(request.form["OnlineBackup_Yes"]),

                float(request.form["DeviceProtection_No_internet_service"]),
                float(request.form["DeviceProtection_Yes"]),

                float(request.form["TechSupport_No_internet_service"]),
                float(request.form["TechSupport_Yes"]),

                float(request.form["StreamingTV_No_internet_service"]),
                float(request.form["StreamingTV_Yes"]),

                float(request.form["StreamingMovies_No_internet_service"]),
                float(request.form["StreamingMovies_Yes"]),

                float(request.form["PaymentMethod_Credit_card"]),
                float(request.form["PaymentMethod_Electronic_check"]),
                float(request.form["PaymentMethod_Mailed_check"]),

                float(request.form["SIM_BSNL"]),
                float(request.form["SIM_Jio"]),
                float(request.form["SIM_Vodafone"]),

                float(request.form["Contract_res"])

            ]

            # Convert to NumPy array
            features = np.array(features).reshape(1, -1)

            # ==========================
            # Prediction
            # ==========================

            pred = model.predict(features)[0]

            # Probability (if supported)
            if hasattr(model, "predict_proba"):

                probability = model.predict_proba(features)[0][1]
                probability = round(probability * 100, 2)

            # Result
            if pred == 1:

                prediction = "Customer is likely to Churn"

            else:

                prediction = "Customer is NOT likely to Churn"

        except Exception as e:

            prediction = f"Error: {str(e)}"
            probability = None

    return render_template(
        "index.html",
        prediction=prediction,
        probability=probability
    )


if __name__ == "__main__":
    app.run(debug=True)