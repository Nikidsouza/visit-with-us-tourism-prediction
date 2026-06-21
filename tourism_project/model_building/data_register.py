import os
from getpass import getpass
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    HF_TOKEN = getpass("Enter HF_TOKEN: ").strip()

if not HF_TOKEN:
    raise ValueError("HF_TOKEN is still empty.")

REPO_ID = "Nikidsouza23/visit-with-us-tourism-prediction"
REPO_TYPE = "dataset"
LOCAL_DATA_DIR = "tourism_project/data"

api = HfApi(token=HF_TOKEN)

try:
    api.repo_info(repo_id=REPO_ID, repo_type=REPO_TYPE)
    print(f"Dataset repo '{REPO_ID}' already exists.")
except RepositoryNotFoundError:
    create_repo(repo_id=REPO_ID, repo_type=REPO_TYPE, private=False, token=HF_TOKEN)
    print(f"Created dataset repo '{REPO_ID}'.")

api.upload_folder(
    folder_path=LOCAL_DATA_DIR,
    repo_id=REPO_ID,
    repo_type=REPO_TYPE,
    commit_message="Upload raw tourism dataset"
)

print("Raw dataset uploaded successfully.")
