# Employee Attrition Prediction

A machine learning web application that predicts whether an employee is likely to leave an organization based on employee-related information.

The project uses a Random Forest classification model and provides both single-employee prediction and bulk CSV prediction through a Flask web application.

## Problem Statement

Employee attrition can negatively affect organizations through increased recruitment costs, loss of experienced employees, and reduced productivity.

The goal of this project is to use machine learning to predict whether an employee is likely to leave the organization based on factors such as job satisfaction, income, overtime, work experience, job involvement, and other employee-related attributes.

## Dataset

The project uses the **IBM HR Analytics Employee Attrition & Performance** dataset.

The dataset contains **1,470 employee records and 35 original attributes** covering employee demographics, job information, compensation, satisfaction, work experience, and other factors related to employee attrition.

The target variable is:

- `Attrition` — whether the employee left the organization (`Yes` or `No`)

Before training the models, four irrelevant constant/identifier columns were removed:

- `EmployeeCount`
- `EmployeeNumber`
- `Over18`
- `StandardHours`

This resulted in **30 input features** used by the prediction model.

## Machine Learning Workflow

The project follows these main steps:

1. Load and inspect the employee dataset.
2. Perform exploratory data analysis (EDA).
3. Remove irrelevant columns.
4. Separate features and the target variable.
5. Split the dataset into training and testing sets using stratified sampling.
6. Encode categorical features using one-hot encoding.
7. Train multiple classification models.
8. Evaluate the models using accuracy, precision, recall, F1-score, and confusion matrices.
9. Select the final model based on its performance for detecting employee attrition.
10. Save the trained model using Joblib.
11. Use the trained model in a Flask web application.
12. Deploy the application on AWS EC2.

## Model Comparison

Three classification models were evaluated during the project:

| Model | Accuracy |
|---|---:|
| Logistic Regression | 71.77% |
| Random Forest | 83.67% |
| HistGradientBoosting | 86.05% |

Although HistGradientBoosting achieved the highest overall accuracy, the final model selected for the application was **Random Forest**.

Random Forest provided a better balance for identifying employees who were likely to leave, which was more important for the project's attrition-prediction objective than accuracy alone.

## Final Model

The final model used in the application is a **Random Forest Classifier**.

### Configuration

```python
RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
## Features

- Single employee attrition prediction
- Attrition probability calculation
- Risk classification:
  - Low Risk
  - Medium Risk
  - High Risk
- Bulk prediction using CSV files
- CSV validation for required columns
- User-friendly error messages
- CSV template download
- Downloadable bulk prediction results
- Flask-based web interface
- AWS EC2 deployment

## Technology Stack

### Programming Language
- Python

### Machine Learning
- Pandas
- NumPy
- Scikit-learn
- Joblib

### Web Development
- Flask
- HTML
- CSS
- JavaScript

### Deployment
- AWS EC2
- Gunicorn
- Nginx
- Git & GitHub

### Development Tools
- Visual Studio Code
- Python Virtual Environment

## Project Structure

```text
Employee-Attrition-Prediction/
│
├── app.py
├── employee_attrition.csv
├── employee_attrition_model.pkl
├── requirements.txt
├── README.md
├── eda.py
├── preprocess.py
├── train_model.py
├── random_forest.py
├── gradient_boosting.py
├── train_final.py
│
├── templates/
│   ├── index.html
│   └── bulk_results.html
│
├── static/
│   └── style.css
│
└── .gitignore

## How to Run Locally

### 1. Clone the Repository

```bash
git clone https://github.com/shah9546/Employee-Attrition-Prediction.git
cd Employee-Attrition-Prediction

python -m venv venv

venv\Scripts\activate

pip install -r requirements.txt

python app.py

http://127.0.0.1:5000

## AWS Deployment

The application is deployed on an Amazon EC2 instance using the following architecture:

Internet
    ↓
Nginx (Port 80)
    ↓
Gunicorn (Port 5000)
    ↓
Flask Application
    ↓
Random Forest Model

### Deployment Components

- **AWS EC2** — Hosts the application.
- **Nginx** — Acts as a reverse proxy and handles HTTP requests.
- **Gunicorn** — Runs the Flask application in production.
- **systemd** — Keeps the application running as a background service.
- **GitHub** — Used for source-code management and deployment updates.

The application is accessible through the EC2 instance's public IP address.

## Disclaimer

This project is developed for educational and demonstration purposes.

The predictions generated by the model are based on the training data and selected employee attributes. They should not be treated as definitive decisions about an employee's future behavior or employment status.