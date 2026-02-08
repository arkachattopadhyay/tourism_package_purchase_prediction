
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import os

# Set your Hugging Face token
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

# Define space details
space_repo_id = "arkac/tourism-package-purchase-prediction-app"
space_type = "space"

api = HfApi(token=os.getenv("HF_TOKEN"))

# Check if space exists, create if not
try:
    api.repo_info(repo_id=space_repo_id, repo_type=space_type)
    print(f"Space '{space_repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Space '{space_repo_id}' not found. Creating new space...")
    create_repo(repo_id=space_repo_id, repo_type=space_type, private=False, space_sdk="streamlit")
    print(f"Space '{space_repo_id}' created.")

# Upload deployment folder to Hugging Face Space
api.upload_folder(
    folder_path="tourism_project/deployment", # the local folder containing your files
    repo_id=space_repo_id,                    # the target repo
    repo_type=space_type,                     # dataset, model, or space
)
print("Deployment files uploaded to Hugging Face Space.")
