import os
import pandas as pd
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi

HF_TOKEN = os.getenv("HF_TOKEN")
DATASET_REPO = "Nikidsouza23/visit-with-us-tourism-prediction" # Corrected repo ID

api = HfApi(token=HF_TOKEN)

LOCAL_DATA_PATH = "tourism_project/data/tourism.csv"
df = pd.read_csv(LOCAL_DATA_PATH)

print("Dataset loaded successfully from local folder")
print("Original shape:", df.shape)

unnamed_cols = [c for c in df.columns if "Unnamed" in str(c)]
if unnamed_cols:
    df.drop(columns=unnamed_cols, inplace=True)

df.drop(columns=["CustomerID"], inplace=True, errors="ignore")
df.drop_duplicates(inplace=True)

for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].astype(str).str.strip()

if "Gender" in df.columns:
    df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})

if "Occupation" in df.columns:
    df["Occupation"] = df["Occupation"].replace({"Free Lancer": "Freelancer"})

if "TypeofContact" in df.columns:
    df["TypeofContact"] = df["TypeofContact"].replace({"Self Inquiry": "Self Enquiry"})

if "MaritalStatus" in df.columns:
    df["MaritalStatus"] = df["MaritalStatus"].replace({"Unmarried": "Single"})
for col in df.select_dtypes(include="number").columns:
    df[col] = df[col].fillna(df[col].median())

for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].fillna(df[col].mode()[0])

target_col = "ProdTaken"

X = df.drop(columns=[target_col])
y = df[target_col]

Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)


files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id=DATASET_REPO, # Corrected repo ID
        repo_type="dataset",
    )
