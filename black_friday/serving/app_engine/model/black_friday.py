import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import joblib
import os
from googleapiclient import discovery
from google.cloud import storage

def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    blob.download_to_filename(destination_file_name)

    print('Blob {} downloaded to {}.'.format(
        source_blob_name,
        destination_file_name))

class BlackFridayModel:

    # Define features and target
    encoder_names = ['User_ID', 'Product_ID', 'Gender', 'Age', 'Occupation', 'City_Category',
                     'Stay_In_Current_City_Years', 'Marital_Status', 'Product_Category_1',
                     'Product_Category_2', 'Product_Category_3']
    features = ['User_ID', 'Product_ID', 'Gender', 'Age', 'Occupation',
                u'City_Category', 'Stay_In_Current_City_Years', 'Marital_Status',
                'Product_Category_1', 'Product_Category_2', 'Product_Category_3']
    categorical_features = ['User_ID', 'Product_ID', 'Gender', 'Age', 'Occupation',
                            'City_Category', 'Marital_Status', 'Product_Category_1',
                            'Product_Category_2', 'Product_Category_3', 'Stay_In_Current_City_Years']

    BUCKET = 'doitintl_black_friday'
    MODEL_VERSION = 'v2'
    model_gcs_path = os.path.join('models/', MODEL_VERSION + "/", 'encoders/')
    encoders_dir = '/tmp/models/encoders'
    os.makedirs(encoders_dir, exist_ok=True)
    models_dir = '/tmp/models/models/'
    os.makedirs(models_dir, exist_ok=True)

    def __init__(self):
        BlackFridayModel.download_encoders()
        self.encoders = {}
        self.load_encoders()


    def load_encoders(self):
        encoders = {}
        for encoder in BlackFridayModel.encoder_names:
            print("Loading encoder to memoery: ", encoder)
            encoders[encoder] = joblib.load(os.path.join(BlackFridayModel.encoders_dir,
                                                              '{}_encoder.joblib'.format(encoder)))
        return encoders

    @staticmethod
    def download_encoders():
        # Download encoders:
        for encod_name in BlackFridayModel.encoder_names:
            encoder_file_name = '{}_encoder.joblib'.format(encod_name)
            gcs_encoder_path = os.path.join(BlackFridayModel.model_gcs_path, encoder_file_name)
            local_encoder_path = os.path.join(BlackFridayModel.encoders_dir, encoder_file_name)
            download_blob(BlackFridayModel.BUCKET, gcs_encoder_path, local_encoder_path)


    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:

        df_train = pd.DataFrame(index=df.index.copy())

        for feature in BlackFridayModel.features:
            print("Transforming features: ", feature)
            if feature in BlackFridayModel.categorical_features:
                vals = self.encoders[feature].transform(df[feature].dropna())
                tmp_srs = df[feature].copy()
                tmp_srs.loc[df[feature].notnull()] = vals
                df_train = df_train.merge(tmp_srs, left_index=True, right_index=True, how='left')
            else:
                df_train[feature] = df[feature].copy()

        return df_train.copy()

