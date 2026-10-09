from flask import Flask, render_template, request, send_file
import pandas as pd
import joblib
import io

# Create Flask application
app = Flask(__name__)

# Load trained machine learning model
model = joblib.load("employee_attrition_model.pkl")

@app.route("/download-template")
def download_template():

    columns = [
        "Age",
        "BusinessTravel",
        "DailyRate",
        "Department",
        "DistanceFromHome",
        "Education",
        "EducationField",
        "EnvironmentSatisfaction",
        "Gender",
        "HourlyRate",
        "JobInvolvement",
        "JobLevel",
        "JobRole",
        "JobSatisfaction",
        "MaritalStatus",
        "MonthlyIncome",
        "MonthlyRate",
        "NumCompaniesWorked",
        "OverTime",
        "PercentSalaryHike",
        "PerformanceRating",
        "RelationshipSatisfaction",
        "StockOptionLevel",
        "TotalWorkingYears",
        "TrainingTimesLastYear",
        "WorkLifeBalance",
        "YearsAtCompany",
        "YearsInCurrentRole",
        "YearsSinceLastPromotion",
        "YearsWithCurrManager"
    ]

    template = pd.DataFrame(columns=columns)

    output = io.BytesIO(
        template.to_csv(index=False).encode("utf-8")
    )

    return send_file(
        output,
        mimetype="text/csv",
        as_attachment=True,
        download_name="employee_attrition_template.csv"
    )


@app.route("/")
def home():
    return render_template("index.html")


# ---------------------------------------------------------
# Single Employee Prediction
# ---------------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

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

    input_data = pd.DataFrame([data])

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1]

    if probability >= 0.70:
        risk_level = "High Risk"
    elif probability >= 0.40:
        risk_level = "Medium Risk"
    else:
        risk_level = "Low Risk"

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


# ---------------------------------------------------------
# Bulk CSV Prediction
# ---------------------------------------------------------

@app.route("/bulk-predict", methods=["POST"])
def bulk_predict():

    file = request.files.get("file")

    if not file:
        return "No CSV file uploaded.", 400

    try:

        # Read uploaded CSV
        df = pd.read_csv(file)

        # Required columns
        required_columns = [
            "Age",
            "BusinessTravel",
            "DailyRate",
            "Department",
            "DistanceFromHome",
            "Education",
            "EducationField",
            "EnvironmentSatisfaction",
            "Gender",
            "HourlyRate",
            "JobInvolvement",
            "JobLevel",
            "JobRole",
            "JobSatisfaction",
            "MaritalStatus",
            "MonthlyIncome",
            "MonthlyRate",
            "NumCompaniesWorked",
            "OverTime",
            "PercentSalaryHike",
            "PerformanceRating",
            "RelationshipSatisfaction",
            "StockOptionLevel",
            "TotalWorkingYears",
            "TrainingTimesLastYear",
            "WorkLifeBalance",
            "YearsAtCompany",
            "YearsInCurrentRole",
            "YearsSinceLastPromotion",
            "YearsWithCurrManager"
        ]

        # Check for missing columns
        missing_columns = [
            column
            for column in required_columns
            if column not in df.columns
        ]

        if missing_columns:
            return (
                "Missing required columns: "
                + ", ".join(missing_columns)
            ), 400

        # Make predictions
        predictions = model.predict(df)

        probabilities = model.predict_proba(df)[:, 1]

        # Add prediction
        df["Prediction"] = [
            "Likely to Leave"
            if p == 1
            else "Likely to Stay"
            for p in predictions
        ]

        # Add probability
        df["Attrition Probability"] = (
            probabilities * 100
        ).round(2)

        # Add risk level
        df["Risk Level"] = [
            "High Risk"
            if p >= 0.70
            else "Medium Risk"
            if p >= 0.40
            else "Low Risk"
            for p in probabilities
        ]

        # Put prediction results first
        result_columns = [
            "Prediction",
            "Attrition Probability",
            "Risk Level"
        ]

        other_columns = [
            column
            for column in df.columns
            if column not in result_columns
        ]

        df = df[result_columns + other_columns]

        # Display results on website
        return render_template(
            "bulk_results.html",
            tables=[
                df.to_html(
                    classes="results-table",
                    index=False
                )
            ]
        )

    except Exception as e:

        return (
            f"Error processing CSV: {str(e)}"
        ), 400


# ---------------------------------------------------------
# Run Flask Application
# ---------------------------------------------------------

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )