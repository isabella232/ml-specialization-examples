from oauth2client.client import GoogleCredentials
from googleapiclient import discovery
from flask import Flask, request
import json
from model.black_friday import BlackFridayModel
import pandas as pd

VERSION_NAME = 'v2'
MODEL_NAME = 'transportation_mode'
PROJECT_ID = 'gad-playground-212407'

app = Flask(__name__)

black_friday_model = BlackFridayModel()


@app.route('/mode_prediction', methods=['POST'])
def generate_predictions():
    """
    Request handler
    :return:
    """
    payload = request.json
    records = payload['gps_trajectories']

    # Extract features from data
    df_raw = pd.Dataframe(records)
    features = BlackFridayModel.extract_features(df_raw)

    # Invokes Cloud ML model
    response = _query_model(features)

    return json.dumps(response)


def _query_model(data):
    """
    Uses a Cloud Machine Learning Engine client to generate predictions
    :param data: tuple of strings with account-billing-ids
    :return: Ordered list on class names ["car", "car" ... ]
    """
    model = 'black_friday'
    project = 'gad-playground-212407'
    version = 'v2'

    instances = data.values.tolist()

    service = discovery.build('ml', 'v1')
    name = 'projects/{}/models/{}'.format(project, model)

    if version is not None:
        name += '/versions/{}'.format('v1')

    response = service.projects().predict(
        name=name,
        body={'instances': instances}
    ).execute()

    if 'error' in response:
        raise RuntimeError(response['error'])

    return response['predictions']


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)