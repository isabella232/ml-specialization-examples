import os
import pandas as pd
from google.cloud import storage
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import joblib

def copy_to_gcs(source, dest):
    bucket = storage.Client().bucket(BUCKET)
    blob = bucket.blob(dest)
    blob.upload_from_filename(source)


BUCKET = 'doitintl_black_friday'
MODEL_VERSION = 'v2'
OPTIMAL_PARAMS = {}


# Crate dirs
print("creating dirs")
models_dir = '/tmp/models/'
os.makedirs(models_dir, exist_ok=True)


# Define features and target
features = ['User_ID', 'Product_ID', 'Gender', 'Age', 'Occupation',
       u'City_Category', 'Stay_In_Current_City_Years', 'Marital_Status',
       'Product_Category_1', 'Product_Category_2', 'Product_Category_3']
categorical_features = ['User_ID', 'Product_ID', 'Gender', 'Age', 'Occupation',
                        'City_Category', 'Marital_Status', 'Product_Category_1',
       'Product_Category_2', 'Product_Category_3', 'Stay_In_Current_City_Years']
label = 'Purchase'


# Read the data
print("Reading file...")
df = pd.read_csv('gs://{BUCKET}/data/BlackFriday.csv'.format(BUCKET=BUCKET))

# Train Encoders
print("Training encoders...")
encoders = {}
for cat_feature in categorical_features:
    print('Fitting categorical feature label encoder to:', cat_feature)
    le = LabelEncoder()
    le.fit(df[cat_feature])
    encoders[cat_feature] = le


# Serialize encoders
encoder_paths = {}
print("Writing encoders...")
transformers_dir = os.path.join(models_dir,'transformers/')
os.makedirs(transformers_dir, exist_ok=True)
for encoder in encoders:
    current_path = os.path.join(transformers_dir, '{}_eocoder.joblib'.format(encoder))
    joblib.dump(encoders[encoder], current_path)
    encoder_paths[encoder] = current_path


# Build Traning set
print("Encoding Features...")
df_train = pd.DataFrame(index=df.index.copy())
for feature in features:
    print(feature)
    if feature in categorical_features:
        vals = encoders[feature].transform(df[feature].dropna())
        tmp_srs = df[feature].copy()
        tmp_srs.loc[df[feature].notnull()] = vals
        df_train = df_train.merge(tmp_srs, left_index=True, right_index=True, how='left')
    else:
        df_train[feature] = df[feature].copy()

dtrain = xgb.DMatrix(df_train[features], label=df[label])

# Train XGBoost model
print('Training the model')
bst = xgb.train(OPTIMAL_PARAMS, dtrain, 20)

# Export the classifier to a file
print('Saving the model')
local_model_path = os.path.join(models_dir, 'model.bst')
bst.save_model(local_model_path)

# Upload all the files to a bucket
print('Uploading transformers to bucket...')
for encoder in encoder_paths:
    dest = os.path.join('models', MODEL_VERSION, 'encoders', encoder + '_encoder.joblib')
    copy_to_gcs(encoder_paths[encoder], dest)

# Copy model
print('Uploading model to bucket...')
dest = os.path.join('models', MODEL_VERSION, 'model', 'model.bst')
copy_to_gcs(local_model_path, dest)

print('Done!')