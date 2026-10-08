from flask import Flask, render_template, request
import pandas as pd
import joblib


# Create Flask application
app = Flask(__name__)


# Load trained machine learning model
model = joblib.load("employee_attrition_model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    # Get values from the HTML form
    data = {
        "Age": int(request.form["Age"]),
        "BusinessTravel": request.form["BusinessTravel"],
        "DailyRate": int(request.form["DailyRate"]),
        "Department": request.form["Department"],
        "DistanceFromHome": int(request.form["DistanceFromHome"]),
        "Education": int(request.form["Education"]),
        "EducationField": request.form["EducationField"],
        "EnvironmentSatisfaction": int(
            request.form["EnvironmentSatisfaction"]
        ),
        "Gender": request.form["Gender"],
        "HourlyRate": int(request.form["HourlyRate"]),
        "JobInvolvement": int(request.form["JobInvolvement"]),
        "JobLevel": int(request.form["JobLevel"]),
        "JobRole": request.form["JobRole"],
        "JobSatisfaction": int(request.form["JobSatisfaction"]),
        "MaritalStatus": request.form["MaritalStatus"],
        "MonthlyIncome": int(request.form["MonthlyIncome"]),
        "MonthlyRate": int(request.form["MonthlyRate"]),
        "NumCompaniesWorked": int(
            request.form["NumCompaniesWorked"]
        ),
        "OverTime": request.form["OverTime"],
        "PercentSalaryHike": int(
            request.form["PercentSalaryHike"]
        ),
        "PerformanceRating": int(
            request.form["PerformanceRating"]
        ),
        "RelationshipSatisfaction": int(
            request.form["RelationshipSatisfaction"]
        ),
        "StockOptionLevel": int(
            request.form["StockOptionLevel"]
        ),
        "TotalWorkingYears": int(
            request.form["TotalWorkingYears"]
        ),
        "TrainingTimesLastYear": int(
            request.form["TrainingTimesLastYear"]
        ),
        "WorkLifeBalance": int(
            request.form["WorkLifeBalance"]
        ),
        "YearsAtCompany": int(
            request.form["YearsAtCompany"]
        ),
        "YearsInCurrentRole": int(
            request.form["YearsInCurrentRole"]
        ),
        "YearsSinceLastPromotion": int(
            request.form["YearsSinceLastPromotion"]
        ),
        "YearsWithCurrManager": int(
            request.form["YearsWithCurrManager"]
        )
    }

    # Convert input into DataFrame
    input_data = pd.DataFrame([data])

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Get prediction probability
    probability = model.predict_proba(input_data)[0][1]

    # Determine risk level
    if probability >= 0.70:
        risk_level = "High Risk"
    elif probability >= 0.40:
        risk_level = "Medium Risk"
    else:
        risk_level = "Low Risk"

    # Convert prediction to readable result
    if prediction == 1:
        result = "Likely to Leave"
    else:
        result = "Likely to Stay"

    return render_template(
        "index.html",
        prediction=result,
        probability=round(probability * 100, 2),
        risk_level=risk_level
    )


if __name__ == "__main__":
    app.run(debug=True)