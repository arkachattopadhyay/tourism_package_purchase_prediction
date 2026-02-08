# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Define constants for the dataset and output paths
repo_id = "arkac/tourism-package-purchase-prediction-data"
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/" + repo_id + "/tourism.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully from Hugging Face.")

# Data cleaning: Remove unnecessary columns
df = df.drop(columns=["Unnamed: 0", "CustomerID"])
print("Unnecessary columns removed.")

# Fix known typos in categorical columns (e.g., Gender)
# Convert 'Fe Male' -> 'Female' to correct dataset typo
if 'Gender' in df.columns:
    df['Gender'] = df['Gender'].astype(str).str.replace(r'Fe\s*Male', 'Female', regex=True).str.strip()
    print("Fixed gender typos if any.")

# Target column
target_col = "ProdTaken"

# Split into X (features) and y (target)
X = df.drop(columns=[target_col])
y = df[target_col]

# Perform train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("Data split into training and testing sets.")

# Save locally
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)
print("Train and test datasets saved locally.")

# Upload back to Hugging Face
files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id=repo_id,
        repo_type="dataset",
    )
print("Processed datasets uploaded back to Hugging Face.")
