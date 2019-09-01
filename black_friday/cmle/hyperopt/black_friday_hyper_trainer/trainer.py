import argparse
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
import hypertune
from google.cloud import storage

parser = argparse.ArgumentParser()
parser.add_argument(
    '--job-dir',
    help='GCS location to write checkpoints and export models',
    required=True,
)
parser.add_argument(
    '--BUCKET_ID',
    help='GCS bucket',
    required=True,
)
parser.add_argument(
    '--max_depth',
    help='max depth for training',
    required=True,
)
parser.add_argument(
    '--learning_rate',
    help='learning rate for training',
    required=True,
)
parser.add_argument(
    '--train-file',
    help='path to training file',
    required=True,
)

def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    blob.download_to_filename(destination_file_name)

    print('Blob {} downloaded to {}.'.format(
        source_blob_name,
        destination_file_name))

args = parser.parse_args()
BUCKET = args.BUCKET_ID
local_train_path = '/tmp/data.csv'

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

download_blob(BUCKET, args.train_file, local_train_path)

# Read the data
print("Reading file...")
df = pd.read_csv(local_train_path)

X_train, X_test, y_train, y_test = train_test_split(df[features].astype(float), df[label].astype(float),
                                                    train_size=0.7, random_state=0)

dtrain = xgb.DMatrix(X_train, label=y_train)
deval = xgb.DMatrix(X_test, label=y_test)

params = {
    'max_depth': args.max_depth,
    'learning_rate': args.learning_rate,
    'objective':'reg:linear',
    'eval_metric':'rmse'
}
bst = xgb.train(params=params, dtrain=dtrain,
                evals=[(deval,'val_1')],
                num_boost_round=10000,
                early_stopping_rounds=100)

rmse = float(bst.eval(deval).split(':')[1])

hpt = hypertune.HyperTune()
hpt.report_hyperparameter_tuning_metric(
    hyperparameter_metric_tag='rmse',
    metric_value=rmse,
    global_step=bst.best_iteration)