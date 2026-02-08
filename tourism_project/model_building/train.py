# for data manipulation
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
# for model training, tuning, and evaluation
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
# for model serialization
import joblib
import os
# for hugging face space authentication to upload files
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError

# Define Repository Id
repo_id = "arkac/tourism-package-purchase-prediction-data"

# Load train and test data from Hugging Face
Xtrain_path = "hf://datasets/" + repo_id + "/Xtrain.csv"
Xtest_path = "hf://datasets/" + repo_id + "/Xtest.csv"
ytrain_path = "hf://datasets/" + repo_id + "/ytrain.csv"
ytest_path = "hf://datasets/" + repo_id + "/ytest.csv"

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path).values.ravel()
ytest = pd.read_csv(ytest_path).values.ravel()
print("Train and test data loaded from Hugging Face.")

# Define features
numeric_features = ['Age', 'CityTier', 'PreferredPropertyStar', 'NumberOfTrips', 'MonthlyIncome', 'DurationOfPitch', 'NumberOfFollowups', 'PitchSatisfactionScore', 'NumberOfPersonVisiting', 'NumberOfChildrenVisiting', 'Passport', 'OwnCar']
categorical_features = ['TypeofContact', 'Occupation', 'Gender', 'ProductPitched', 'MaritalStatus', 'Designation']

# Preprocessing pipeline
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)

# Define XGBoost Classifier
xgb_model = xgb.XGBClassifier(random_state=42, objective="binary:logistic")

# Define hyperparameter grid (for use with pipeline, prefix with model name)
param_grid = {
    'xgbclassifier__n_estimators': [50, 100],
    'xgbclassifier__max_depth': [3, 5],
    'xgbclassifier__learning_rate': [0.01, 0.1],
    'xgbclassifier__subsample': [0.8, 1.0],
}

# Create pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)

# Grid search with cross-validation
grid_search = GridSearchCV(
    model_pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1
)
grid_search.fit(Xtrain, ytrain)

# Best model
best_model = grid_search.best_estimator_
print("Best Parameters:\n", grid_search.best_params_)

# Predictions
y_pred_train = best_model.predict(Xtrain)
y_pred_test = best_model.predict(Xtest)

# Evaluation
print("\nTraining Performance:")
print("Accuracy:", accuracy_score(ytrain, y_pred_train))

print("\nTest Performance:")
print("Accuracy:", accuracy_score(ytest, y_pred_test))
print("Classification Report:")
print(classification_report(ytest, y_pred_test))

# Save best model
joblib.dump(best_model, "best_tourism_package_purchase_prediction_model.joblib")
print("Model saved as best_tourism_package_purchase_prediction_model.joblib")

# Upload to Hugging Face Model Hub
model_repo_id = "arkac/tourism-package-purchase-prediction-model"
model_repo_type = "model"

api = HfApi(token=os.getenv("HF_TOKEN"))

# Step 1: Check if the model space exists, create if not
try:
    api.repo_info(repo_id=model_repo_id, repo_type=model_repo_type)
    print(f"Model Space '{model_repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Model Space '{model_repo_id}' not found. Creating new Space...")
    create_repo(repo_id=model_repo_id, repo_type=model_repo_type, private=False)
    print(f"Model Space '{model_repo_id}' created.")

# Upload model
api.upload_file(
    path_or_fileobj="best_tourism_package_purchase_prediction_model.joblib",
    path_in_repo="best_tourism_package_purchase_prediction_model.joblib",
    repo_id=model_repo_id,
    repo_type=model_repo_type,
)
print("Best model uploaded to Hugging Face Model Hub.")
