import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import mlflow
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

# Define Hugging Face repository IDs and token
HF_DATASET_REPO = "Nikidsouza23/visit-with-us-tourism-prediction"
HF_MODEL_REPO = "Nikidsouza23/visit-with-us-tourism-model" # Defined for model upload
HF_TOKEN = os.getenv("HF_TOKEN")

# Initialize HfApi client
api = HfApi(token=HF_TOKEN)

# Set MLflow tracking URI and experiment name
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("visit-with-us-tourism-classification") # Changed experiment name

# Define paths to the split datasets on Hugging Face
Xtrain_path = f"hf://datasets/{HF_DATASET_REPO}/Xtrain.csv"
Xtest_path  = f"hf://datasets/{HF_DATASET_REPO}/Xtest.csv"
ytrain_path = f"hf://datasets/{HF_DATASET_REPO}/ytrain.csv"
ytest_path  = f"hf://datasets/{HF_DATASET_REPO}/ytest.csv"

# Load data directly from Hugging Face
X_train = pd.read_csv(Xtrain_path)
X_test = pd.read_csv(Xtest_path)
y_train = pd.read_csv(ytrain_path).squeeze()
y_test = pd.read_csv(ytest_path).squeeze()

print("Train and test datasets loaded successfully from Hugging Face.")

# Dynamically determine numeric and categorical features from the loaded X_train
numeric_features = X_train.select_dtypes(include="number").columns.tolist()
categorical_features = X_train.select_dtypes(include="object").columns.tolist()

# Preprocessing pipelines for numeric and categorical features
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy="median")),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy="most_frequent")),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Create a column transformer to apply different transformations to different columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Define base XGBoost Classifier
xgb_model = XGBClassifier(random_state=42, n_jobs=-1, eval_metric="logloss")

# Hyperparameter grid for XGBClassifier (adjusted from Zk7oOpa5xrya)
param_grid = {
    'xgbclassifier__n_estimators': [100, 200],
    'xgbclassifier__max_depth': [3, 5, 7],
    'xgbclassifier__learning_rate': [0.05, 0.1],
    'xgbclassifier__subsample': [0.8, 1.0],
    'xgbclassifier__colsample_bytree': [0.8, 1.0]
}

# Create the full pipeline with preprocessor and XGBClassifier
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('xgbclassifier', xgb_model)
])

with mlflow.start_run():
    # Grid Search
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=3, n_jobs=-1, scoring='f1', verbose=1) # Scoring 'f1' for classification
    grid_search.fit(X_train, y_train)

    # Log parameter sets from GridSearchCV results
    results = grid_search.cv_results_
    for i in range(len(results['params'])):
        param_set = results['params'][i]
        mean_score = results['mean_test_score'][i]

        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("mean_f1_score", mean_score) # Changed metric name

    # Best model
    mlflow.log_params(grid_search.best_params_)
    best_model = grid_search.best_estimator_

    # Predictions
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    # Metrics for classification (updated from regression metrics)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob)
    }

    # Log metrics
    mlflow.log_metrics(metrics)
    print("Classification metrics logged to MLflow:", metrics)

    # Save and log the best model artifact locally
    model_local_path = "best_tourism_model.joblib" # Changed model name
    joblib.dump(best_model, model_local_path)
    mlflow.log_artifact(model_local_path, artifact_path="model")
    print(f"Model saved locally and logged as artifact at: {model_local_path}")

    # Upload to Hugging Face Model Hub
    # Step 1: Check if the model repo exists
    try:
        api.repo_info(repo_id=HF_MODEL_REPO, repo_type="model")
        print(f"Model repository '{HF_MODEL_REPO}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"Model repository '{HF_MODEL_REPO}' not found. Creating new repository...")
        create_repo(repo_id=HF_MODEL_REPO, repo_type="model", private=False, token=HF_TOKEN)
        print(f"Model repository '{HF_MODEL_REPO}' created.")

    api.upload_file(
        path_or_fileobj=model_local_path,
        path_in_repo="best_tourism_model.joblib", # Consistent filename in repo
        repo_id=HF_MODEL_REPO,
        repo_type="model",
    )
    print(f"Best model uploaded successfully to Hugging Face Model Hub: {HF_MODEL_REPO}")

print("Model training and Hugging Face upload process complete.")
