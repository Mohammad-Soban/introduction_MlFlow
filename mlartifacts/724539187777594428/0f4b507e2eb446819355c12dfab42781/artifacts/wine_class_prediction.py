import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# MLFlow URL
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Load the wine dataset
wine_data = load_wine()
x = wine_data.data
y = wine_data.target


# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# Defining the parameters of the Random Forest Classifier
n_estimators = 5
max_depth = 5


mlflow.set_experiment("Wine-Data-RF-Experiment")

# Create and train the Random Forest Classifier
with mlflow.start_run():

    rf_classifier = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    rf_classifier.fit(x_train, y_train)

    # Make predictions on the test set
    y_pred = rf_classifier.predict(x_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    # Print classification report
    print(classification_report(y_test, y_pred))

    # Log the model and metrics to MLflow
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=wine_data.target_names, yticklabels=wine_data.target_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")
    mlflow.log_artifact(__file__)

    mlflow.set_tags({
        "model": "Random Forest Classifier",
        "dataset": "Wine Dataset",
        "author": "Soban",
        "version": "1.0",
        "Project": "Wine Class Prediction"
    })

    mlflow.sklearn.log_model(rf_classifier, "Model")