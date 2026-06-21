import os
import joblib
import mlflow
import pandas as pd
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN is not set.")

HF_DATASET_REPO = "Nikidsouza23/visit-with-us-tourism-prediction"
HF_MODEL_REPO = "Nikidsouza23/visit-with-us-tourism-model"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
MODEL_LOCAL_PATH = "best_tourism_model.joblib"

api = HfApi(token=HF_TOKEN)
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("visit-with-us-tourism-classification")

X_train = pd.read_csv(f"hf://datasets/{HF_DATASET_REPO}/Xtrain.csv")
X_test = pd.read_csv(f"hf://datasets/{HF_DATASET_REPO}/Xtest.csv")
y_train = pd.read_csv(f"hf://datasets/{HF_DATASET_REPO}/ytrain.csv").squeeze("columns")
y_test = pd.read_csv(f"hf://datasets/{HF_DATASET_REPO}/ytest.csv").squeeze("columns")

numeric_features = X_train.select_dtypes(include="number").columns.tolist()
categorical_features = X_train.select_dtypes(include="object").columns.tolist()

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("xgbclassifier", XGBClassifier(random_state=42, n_jobs=-1, eval_metric="logloss"))
])

param_grid = {
    "xgbclassifier__n_estimators": [100, 200],
    "xgbclassifier__max_depth": [3, 5, 7],
    "xgbclassifier__learning_rate": [0.05, 0.1],
    "xgbclassifier__subsample": [0.8, 1.0],
    "xgbclassifier__colsample_bytree": [0.8, 1.0]
}

with mlflow.start_run(run_name="xgboost_gridsearch"):
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=3,
        scoring="f1",
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }

    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metrics(metrics)

    joblib.dump(best_model, MODEL_LOCAL_PATH)
    mlflow.log_artifact(MODEL_LOCAL_PATH, artifact_path="model")

    print("Best params:", grid_search.best_params_)
    print("Metrics:", metrics)

try:
    api.repo_info(repo_id=HF_MODEL_REPO, repo_type="model")
except RepositoryNotFoundError:
    create_repo(repo_id=HF_MODEL_REPO, repo_type="model", private=False, token=HF_TOKEN)

api.upload_file(
    path_or_fileobj=MODEL_LOCAL_PATH,
    path_in_repo="best_tourism_model.joblib",
    repo_id=HF_MODEL_REPO,
    repo_type="model",
    commit_message="Upload best tourism model"
)

print(f"Model uploaded successfully to {HF_MODEL_REPO}")
