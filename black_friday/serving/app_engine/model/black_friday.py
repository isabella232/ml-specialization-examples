import pandas as pd
import numpy as np

class GPSTrajectoriesModel:

    FEATURES = []

    @staticmethod
    def extract_features(data: dict) -> pd.DataFrame:

        df_raw = pd.DataFrame().from_dict(data)
        features = GPSTrajectoriesModel.extraction_logics(df_raw)
        df_features = pd.DataFrame().from_dict(features, orient='index').T
        return df_features[GPSTrajectoriesModel.FEATURES].values.tolist()

    @staticmethod
    def extraction_logics(df: pd.DataFrame) -> dict:
        """
        Extracts features from a DataFrame representing a gps trajectory
        :param df:
        :return:
        """
        features = {}

        # Calculate the speed
        df = GPSTrajectoriesModel.calc_speed(df)

        # meta
        features['segment'] = df['segment'].iloc[0]

        # location
        features['latitude_mean'] = df['latitude'].mean()
        features['longitude_mean'] = df['longitude'].mean()
        features['latitude_std'] = df['latitude'].std()
        features['longitude_std'] = df['longitude'].std()

        # velocity
        hist, _ = np.histogram(df['speed'], bins=range(0, 200, 5), normed=True)
        for i in range(0, int(195 / 5)):
            features['velocity_%d' % (i * 5)] = hist[i]

        # sample freq
        features['time_diff_mean'] = df['time_diff'].mean()
        features['time_diff_median'] = df['time_diff'].median()
        features['time_diff_std'] = df['time_diff'].std()

        return features

