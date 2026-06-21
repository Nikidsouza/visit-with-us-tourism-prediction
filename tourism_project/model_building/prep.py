import os
from getpass import getpass
import pandas as pd
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    HF_TOKEN = getpass("Enter HF_TOKEN: ").strip()

if not HF_TOKEN:
    raise ValueError("HF_TOKEN is still empty.")

DATASET_REPO = "Nikidsouza23/visit-with-us-tourism-prediction"
LOCAL_DATA_PATH = "tourism_project/data/tourism.csv"
OUTPUT_DIR = "tourism_project/data/processed"

os.makedirs(OUTPUT_DIR, exist_ok=True)
api = HfApi(token=HF_TOKEN)

try:
    api.repo_info(repo_id=DATASET_REPO, repo_type="dataset")
except RepositoryNotFoundError:
    create_repo(repo_id=DATASET_REPO, repo_type="dataset", private=False, token=HF_TOKEN)

df = pd.read_csv(LOCAL_DATA_PATH)
print("Dataset loaded successfully.")
print("Original shape:", df.shape)

unnamed_cols = [c for c in df.columns if "Unnamed" in str(c)]
if unnamed_cols:
    df.drop(columns=unnamed_cols, inplace=True)

df.drop(columns=["CustomerID"], inplace=True, errors="ignore")
df.drop_duplicates(inplace=True)

for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].astype(str).str.strip()

replacements = {
    "Gender": {"Fe Male": "Female"},
    "Occupation": {"Free Lancer": "Freelancer"},
    "TypeofContact": {"Self Inquiry": "Self Enquiry"},
    "MaritalStatus": {"Unmarried": "Single"},
}
for col, mapping in replacements.items():
    if col in df.columns:
        df[col] = df[col].replace(mapping)

for col in df.select_dtypes(include="number").columns:
    df[col] = df[col].fillna(df[col].median())

for col in df.select_dtypes(include="object").columns:
    mode_val = df[col].mode(dropna=True)
    if not mode_val.empty:
        df[col] = df[col].fillna(mode_val.iloc[0])

target_col = "ProdTaken"
X = df.drop(columns=[target_col])
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

paths = {
    "Xtrain.csv": os.path.join(OUTPUT_DIR, "Xtrain.csv"),
    "Xtest.csv": os.path.join(OUTPUT_DIR, "Xtest.csv"),
    "ytrain.csv": os.path.join(OUTPUT_DIR, "ytrain.csv"),
    "ytest.csv": os.path.join(OUTPUT_DIR, "ytest.csv"),
}

X_train.to_csv(paths["Xtrain.csv"], index=False)
X_test.to_csv(paths["Xtest.csv"], index=False)
y_train.to_frame(name=target_col).to_csv(paths["ytrain.csv"], index=False)
y_test.to_frame(name=target_col).to_csv(paths["ytest.csv"], index=False)

for remote_name, local_path in paths.items():
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=remote_name,
        repo_id=DATASET_REPO,
        repo_type="dataset",
        commit_message=f"Upload {remote_name}"
    )

print("Train/test splits uploaded successfully.")
