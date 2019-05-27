import argparse
import datetime
import os
import subprocess
import sys
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
import hypertune
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import joblib

parser = argparse.ArgumentParser()
parser.add_argument(
    '--job-dir',
    help='GCS location to write checkpoints and export models',
    required=True,
)

parser.add_argument(
    '--num-boost-round',
    help='Number of boosting iterations.',
    type=int,
    default=10
)

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
transformers_dir = os.path.join(models_dir, 'transformers/')
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

X_train, X_test, y_train, y_test = train_test_split(df_train[features], df_train[label],
                                                    train_size=0.7, 4, random_state=0)
dtrain = xgb.DMatrix(X_train, label=y_train)

bst = xgb.train(params={}, dtrain=dtrain, num_boost_round=args.num_boost_round)

deval = xgb.DMatrix(X_test, label=y_test)
rmse = float(bst.eval(deval).split(':')[1])

hpt = hypertune.HyperTune()
hpt.report_hyperparameter_tuning_metric(
    hyperparameter_metric_tag='my_metric_tag',
    metric_value=rmse,
    global_step=1)