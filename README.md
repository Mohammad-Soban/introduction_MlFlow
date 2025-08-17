## ML FLOW
#### This repository contains the complete tutorial for Machine Learning Operations (MLOps) using ML Flow.


### Why ML Flow over DVC?
- DVC is primarily used along with Git for version control of data and models, while ML Flow is a more comprehensive platform that provides tools for managing the entire machine learning lifecycle.
- MLFlow's core strength lies in managing the entire machine learning lifecycle, with a heavy emphasis on experiment tracking. Its components are designed to streamline the process from initial experimentation to production deployment.
- DVC DVC (Data Version Control) is primarily focused on versioning data and machine learning pipelines. It is often described as "Git for data."
- ML Flow is preferred more due to its Holistic Approach to MLOps, a better UI for collaborative work, model management, and deployment capabilities.


### How to use ML Flow?

1. **Install ML Flow**: You can install ML Flow using pip:
    ```bash
    pip install mlflow
    ```

2. **Start ML Flow Server**: You can start the ML Flow server to track experiments:
    ```bash
    mlflow ui
    ```

3. **Start Tracking**: In your Python script, you can start tracking experiments using ML Flow:
    ```python
    import mlflow
    mlflow.start_run()
    ```

    OR

    ```python
    with mlflow.start_run():
        # Your training code here
    ```

4. **Log Parameters and Metrics**: You can log parameters, metrics, and artifacts during your training process:
    ```python
    mlflow.log_param("param_name", param_value)
    mlflow.log_metric("metric_name", metric_value)
    mlflow.log_artifact("path/to/artifact")
    ```

5. **View Experiments**: After running your script, you can view the logged experiments by navigating to `http://localhost:5000` in your web browser.

6. **Model Registry**: You can register your models in the ML Flow Model Registry for versioning and deployment:
    ```python
    mlflow.register_model("runs:/<run_id>/model", "model_name")
    ```

7. **Deploy Models**: You can deploy your models using ML Flow's deployment capabilities:
    ```bash
    mlflow models serve -m runs:/<run_id>/model
    ```


### What things can actually be logged in ML FLOW?
### 1. **Metrics:**
   - **Accuracy**: Track model accuracy over different runs.
   - **Loss**: Log training and validation loss during the training process.
   - **Precision, Recall, F1-Score**: Log evaluation metrics for classification tasks.
   - **AUC (Area Under Curve)**: Track AUC for classification models.
   - **Custom Metrics**: Any numeric value can be logged as a custom metric (e.g., RMSE, MAE).

### 2. **Parameters:**
   - **Model Hyperparameters**: Log values such as learning rate, number of trees, max depth, etc.
   - **Data Processing Parameters**: Track parameters used in data preprocessing, such as the ratio of train-test split or feature selection criteria.
   - **Feature Engineering**: Log any parameters related to feature extraction or engineering.

### 3. **Artifacts:**
   - **Trained Models**: Save and version models for easy retrieval and comparison.
   - **Model Summaries**: Log model summaries or architecture details.
   - **Confusion Matrices**: Save visualizations of confusion matrices.
   - **ROC Curves**: Log Receiver Operating Characteristic curves.
   - **Plots**: Save any custom plots like loss curves, feature importances, etc.
   - **Input Data**: Log the datasets used in training and testing.
   - **Scripts & Notebooks**: Save code files or Jupyter notebooks used in the experiment.
   - **Environment Files**: Track environment files like `requirements.txt` or `conda.yaml` to ensure reproducibility.

### 4. **Models:**
   - **Pickled Models**: Log models in a serialized format that can be reloaded later.
   - **ONNX Models**: Log models in the ONNX format for cross-platform usage.
   - **Custom Models**: Log custom models using MLflow’s model interface.

### 5. **Tags:**
   - **Run Tags**: Tag your experiments with metadata like author name, experiment description, or model type.
   - **Environment Tags**: Tag with environment-specific details like `gpu` or `cloud_provider`.

### 6. **Source Code:**
   - **Scripts**: Track the script or notebook used in the experiment.
   - **Git Commit**: Log the Git commit hash to link the experiment with a specific version of the code.
   - **Dependencies**: Track the exact version of libraries and dependencies used.

### 7. **Logging Inputs and Outputs:**
   - **Training Data**: Log the training data used in the experiment.
   - **Test Data**: Log the test or validation datasets.
   - **Inference Outputs**: Track the predictions or outputs of the model on a test set.

### 8. **Custom Logging:**
   - **Custom Objects**: Log any Python object or file type as a custom artifact.
   - **Custom Functions**: Track custom functions or methods used within the experiment.

### 9. **Model Registry:**
   - **Model Versioning**: Track different versions of models and their lifecycle stages (e.g., `Staging`, `Production`).
   - **Model Deployment**: Manage and track the deployment status of different models.

### 10. **Run and Experiment Details:**
   - **Run ID**: Each run is assigned a unique identifier.
   - **Experiment Name**: Group multiple runs under a single experiment name.
   - **Timestamps**: Log start and end times of each run to track duration.


### Important Note:
- When tracking the file using the keyword __file__, the MLFLow server treats the tracking uri as a file:// object and not http or https. Hence it throws an error.
- Hence we need to first set the tracking uri to a file path using the following command:
    ```python
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    ```
- So the error wont occur now.


### Creating New Experiment:
- We can create a new experiment using two methods:
    1. Using the UI at the ML Flow server.
    
    2. Using the following command in the code:
    ```python
    mlflow.create_experiment("experiment_name")
    ```


### Setting The Experiment:
- We can set the experiment using the either by mentioning it using the set_experiment method or by passing the experiment_id as an argument in the start_run method.
    ```python
    mlflow.set_experiment("experiment_name")
    ```

    OR

    ```python
    with mlflow.start_run(experiment_id="experiment_id"):
        # Your training code here
    ```


### The Difference Between Experiments and Runs
- Experiments are collections of runs that share a common purpose or goal, while runs are individual executions of a machine learning workflow within an experiment.
- An experiment can have multiple runs, each representing a different configuration, hyperparameters, or data splits.
- Runs are tracked with unique identifiers and can be compared against each other within the context of the experiment.
- For example, three people are trying to solve a Prediction Problem. One of them is using a Decision Tree, the other is using a Random Forest, and the third is using a Neural Network. So each of them can create an experiment and try different hyperparameters and configurations for their respective models. Each of their runs can be tracked under the same experiment, allowing for easy comparison of results.


## Connection With DagsHub

### What is DagsHub
- DagsHub is a platform for managing machine learning projects, providing tools for version control, data management, and collaboration. It integrates with popular ML frameworks and tools, making it easier to track experiments, datasets, and models in a single place.
- While working with ML-Flow we only have the access of MLFLow in the local, but to connect it to cloud we use DagsHub.
- It can also be done on AWS, but for that stuff we need to first create an IAM User, an EC2 instance and a storage as well for storing the artifacts, models and Datasets and then integrate it with ML-FLow for tracking.
- It is long process and hence DagsHub is used, we just need to connect the github repo and get the desired results in ML-FLow. Hence Dagshub is a more convenient option for cloud integration.


### Connection With DagsHub
- Signin/Signup on Dagshub using Github (Preferred)
- Go to dagshub.com/repo/connect and connect your repo which you want to track with ML-Flow.
- Then a github like UI will open and then under the remote button, go to experiments and copy the link and save it somewhere.
- Also there is a piece of code under MLFLow - Tracking which needs to be copied and saved


### Installation Of Dagshub
- Before using Dagshub, we need to install it using the following command:
```bash
pip install dagshub
```


### Initialization Of Dagshub
- After installing Dagshub, we need to initialize it in our project. This can be done using the following code:
```python
import dagshub
dagshub.init(repo_owner='Mohammad-Soban', repo_name='introduction_MlFlow', mlflow=True)
```


### Setting up the Remote URL For Tracking
- After initialization we need to set a remote URL for tracking, and this can be done using the following code:
```python
mlflow.set_tracking_uri("The link copied From Dagshub")
```
- The link in our case is <b><u>https://dagshub.com/Mohammad-Soban/introduction_MlFlow.mlflow</u></b>


### AUTO-LOGGING 
- For enabling the auto-logging feature in MLflow, you can use the following code snippet:
```python
mlflow.autolog()
```
- This will automatically log all the relevant parameters, metrics, and models without the need for explicit logging calls.
- Only the code, if needed to log should be explicitly logged using the mlflow.log_* functions.
- But autologger sometimes logs a lot of things which are not always useful, so we generally use the mlflow.log_* functions for more control over what gets logged.
- **`mlflow.autolog()`** is a powerful feature in MLflow that automatically logs parameters, metrics, models, and other relevant information during your machine learning training process. However, it's important to know what can and cannot be logged automatically.

### Things That Can Be Logged by `mlflow.autolog`
1. **Parameters:**
   - Hyperparameters used to train the model, such as `max_depth`, `learning_rate`, `n_estimators`, etc.

2. **Metrics:**
   - Common evaluation metrics like accuracy, precision, recall, and loss values, depending on the model and framework being used.

3. **Model:**
   - The trained model itself is automatically logged.

4. **Artifacts:**
   - Certain artifacts like model summary and plots (e.g., learning curves, confusion matrix) are logged if supported by the framework.

5. **Framework-Specific Information:**
   - Framework-specific details like early stopping criteria in gradient boosting models or deep learning models (e.g., number of epochs, optimizer configuration).

6. **Environment Information:**
   - Details about the environment such as installed libraries and versions.

7. **Training Data and Labels:**
   - Information about the dataset size and sometimes feature information, but not the entire dataset itself.

8. **Automatic Model Signature:**
   - Autologging can infer the input types (signature) of the model and save them along with the model.

### Things That Cannot Be Logged by `mlflow.autolog`    
1. **Custom Metrics:**
   - Metrics not included in the default set for the specific framework (e.g., F1 score if it's not the default metric) will not be logged unless manually specified.

2. **Custom Artifacts:**
   - Custom plots, charts, or files that are not part of the default model training process (e.g., a custom visualization or report).

3. **Preprocessed Data:**
   - The transformed or preprocessed data used during training or testing is not logged unless you manually log it as an artifact.

4. **Intermediate Model States:**
   - Models saved at intermediate stages of training (e.g., after every epoch) are not logged unless explicitly done so.

5. **Complex Model Structures:**
   - If you're using a non-standard or highly customized model structure, `mlflow.autolog` might miss some logging details.

6. **Non-standard Training Loops:**
   - If your training loop is not compatible with the standard loops expected by MLflow (e.g., custom training loops), autologging might not capture everything correctly.

7. **Non-Supported Frameworks:**
   - `mlflow.autolog` does not support all frameworks. If your model is built with a framework that MLflow doesn’t support, autologging won’t work.

8. **Custom Hyperparameter Tuning:**
   - Hyperparameters or configurations that are outside the scope of the framework’s autologging capabilities (e.g., specific settings in a custom grid search).


### Summary
- **Use Cases:** `mlflow.autolog` is great for quick and convenient logging, especially for standard workflows in supported frameworks.
- **Limitations:** Custom elements, complex structures, and unsupported frameworks require manual logging to capture all relevant details.
