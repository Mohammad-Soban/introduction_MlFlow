from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
import mlflow
import dagshub
import pandas as pd
import joblib
import os

# Setting the mlflow tracking URI
mlflow.set_tracking_uri("http://localhost:5000")

# Loading the Dataset
dataset = load_breast_cancer()
x = pd.DataFrame(dataset.data, columns=dataset.feature_names)
y = pd.Series(dataset.target)

# Split the Dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Create an object of the Random Forest Classifier
rf = RandomForestClassifier(random_state=42)

# Creating a parameters grid
params = {
    'n_estimators': [10, 20, 50, 100],
    'max_depth': [None, 10, 15, 20, 30]
}

# Applying GridSearch CV to find the best parameters
grid_search = GridSearchCV(estimator=rf, param_grid=params, cv=5, n_jobs=-1, verbose=2)

# Applying MLFLow to track our experiments and metrices
with mlflow.start_run():

    # Fitting the model
    grid_search.fit(x_train, y_train)

    # Displyaing the best params and the best score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    # Logging the parameters and the score
    mlflow.log_params(best_params)
    mlflow.log_metric("best_score", best_score)

    # Logging the Training Data
    train_df = x_train.copy()
    train_df['target'] = y_train
    train_df = mlflow.data.from_pandas(train_df)
    mlflow.log_input(train_df, "Train_Data")

    # Logging the Test Data
    test_df = x_test.copy()
    test_df['target'] = y_test
    test_df = mlflow.data.from_pandas(test_df)
    mlflow.log_input(test_df, "Test_Data")

    # Logging the Source Code
    mlflow.log_artifact(__file__)

    # Logging the Best Model
    best_model = grid_search.best_estimator_
    
    joblib.dump(best_model, "best_model.pkl")
    mlflow.log_artifact("best_model.pkl")
    # OR
    mlflow.sklearn.log_model(best_model, "Random_Forest")

    # Setting the tags
    mlflow.set_tags({
        "model": "Random_Forest",
        "dataset": "Breast Cancer",
        "version": "1.0"
    })