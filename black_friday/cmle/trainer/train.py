import os
import pandas as pd
from google.cloud import storage
import xgboost as xgb
import argparse

def copy_to_gcs(source, dest):
    bucket = storage.Client().bucket(BUCKET)
    blob = bucket.blob(dest)
    blob.upload_from_filename(source)

parser = argparse.ArgumentParser()
parser.add_argument(
    '--BUCKET',
    help='GCS location to write checkpoints and export models',
    required=True,
)

parser.add_argument(
    '--MODEL_VERSION',
    help='name of the model version to create',
    type=int,
    default=10
)

args = parser.parse_args()
BUCKET = args.BUCKET
MODEL_VERSION = args.MODEL_VERSION
OPTIMAL_PARAMS = {}


# Crate dirs
print("creating dirs")
models_dir = '/tmp/models/'
os.makedirs(models_dir, exist_ok=True)


# Define features and target
features = ['product_id',
 'gender',
 'age',
 'occupation',
 'city_category',
 'stay_in_current_city_years',
 'marital_status',
 'product_category_1',
 'product_category_2',
 'product_category_3']

label = 'Purchase'

# Read the data
print("Reading file...")
df = pd.read_csv('gs://{BUCKET}/data/train_data.csv/part-00000-6a527ad3-ad67-4928-86a7-9d568115b70f-c000.csv'.format(BUCKET=BUCKET))

dtrain = xgb.DMatrix(df[features], label=df[label])

# Train XGBoost model
print('Training the model')
bst = xgb.train(OPTIMAL_PARAMS, dtrain, 200)

# Export the classifier to a file
print('Saving the model')
local_model_path = os.path.join(models_dir, 'model.bst')
bst.save_model(local_model_path)

# Copy model
print('Uploading model to bucket...')
dest = os.path.join('models', MODEL_VERSION, 'model', 'model.bst')
copy_to_gcs(local_model_path, dest)

print('Done!')