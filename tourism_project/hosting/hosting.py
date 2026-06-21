import os
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN is not set.")

SPACE_REPO = "Nikidsouza23/visit-with-us-tourism-app"
SPACE_SDK = "streamlit"
LOCAL_DEPLOYMENT_DIR = "tourism_project/deployment"

api = HfApi(token=HF_TOKEN)

try:
    api.repo_info(repo_id=SPACE_REPO, repo_type="space")
    print(f"Space '{SPACE_REPO}' already exists.")
except RepositoryNotFoundError:
    create_repo(
        repo_id=SPACE_REPO,
        repo_type="space",
        private=False,
        space_sdk=SPACE_SDK,
        token=HF_TOKEN
    )
    print(f"Created Space '{SPACE_REPO}'.")

api.upload_folder(
    folder_path=LOCAL_DEPLOYMENT_DIR,
    repo_id=SPACE_REPO,
    repo_type="space",
    commit_message="Deploy Streamlit app to Hugging Face Space"
)

print(f"Deployment files uploaded successfully to Space: {SPACE_REPO}")
