from google.cloud import bigquery
from datetime import datetime
from googleapiclient import discovery
import time
import pandas as pd
import argparse

JOB_RUNNING_STATES = ['QUEUED', 'PREPARING', 'RUNNING']
JOB_COMPLETED_STATES = ['SUCCEEDED']
JOB_FAILED_STATES = ['FAILED', 'CANCELLING', 'CANCELLED', 'STATE_UNSPECIFIED']

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--project_id',
        help='Google Cloud Project ID',
        required=True,
        type=str
    )

    parser.add_argument(
        '--dataset_id',
        help='Name of BigQuery Dataset ID to save training results',
        type=str,
    )

    parser.add_argument(
        '--model_name',
        help='name of the AI platform model that will be created',
        required=True,
        type=str

    )

    parser.add_argument(
        '--model_version',
        help='name of the AI platform model version that will be created',
        required=True,
        type=str

    )

    parser.add_argument(
        '--bucket_name',
        help='Name of the staging bucket',
        required=True,
        type=str

    )

    parser.add_argument(
        '--create_model',
        help='create a new model? default is False',
        required=False,
        type=bool

    )
    args = parser.parse_args()


    return args

def create_train():
    client = bigquery.Client()
    query = """
        WITH dataset AS( SELECT 

              EXTRACT(HOUR FROM  trip_start_timestamp) trip_start_hour
            , EXTRACT(DAYOFWEEK FROM  trip_start_timestamp) trip_start_weekday
            , EXTRACT(WEEK FROM  trip_start_timestamp) trip_start_week
            , EXTRACT(DAYOFYEAR FROM  trip_start_timestamp) trip_start_yearday
            , EXTRACT(MONTH FROM  trip_start_timestamp) trip_start_month
            , (trip_miles * 1.60934 ) / ((trip_seconds + .01) / (60 * 60)) trip_speed_kmph
            , trip_miles
            , pickup_latitude
            , pickup_longitude
            , dropoff_latitude
            , dropoff_longitude
            , pickup_community_area
            , dropoff_community_area
            , ST_DISTANCE(
              (ST_GEOGPOINT(pickup_longitude,pickup_latitude)),
              (ST_GEOGPOINT(dropoff_longitude,dropoff_latitude))) air_distance
            , CAST (trip_seconds AS FLOAT64) trip_seconds
        FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips` 
            WHERE RAND() < (3000000/112860054) --sample maximum ~3M records 
                    AND  trip_start_timestamp < '2016-01-01'
                    AND pickup_location IS NOT NULL
                    AND dropoff_location IS NOT NULL)
        SELECT 
             trip_seconds
            , air_distance
            , pickup_latitude
            , pickup_longitude
            , dropoff_latitude
            , dropoff_longitude
            , pickup_community_area
            , dropoff_community_area
            , trip_start_hour
            , trip_start_weekday
            , trip_start_week
            , trip_start_yearday
            , trip_start_month
        FROM dataset
        WHERE trip_speed_kmph BETWEEN 5 AND 90
    """
    job_config = bigquery.QueryJobConfig()
    table_ref = client.dataset(dataset_id).table(table_name)
    job_config.destination = table_ref
    sql = query

    query_job = client.query(
        sql,
        location='US',
        job_config=job_config)

    query_job.result()
    print('Query results loaded to table {}'.format(table_ref.path))

    return table_ref


def export_training_to_gcs(table_ref):
    """
    Exporting the dataset table to GCS
    :param table_ref: the table to export
    :return:
    """
    client = bigquery.Client()
    destination_uri = data_dir
    job_config = bigquery.ExtractJobConfig(print_header=False)
    extract_job = client.extract_table(
        table_ref,
        destination_uri,
        location="US",
        job_config=job_config)
    extract_job.result()

    print(
        "Exported {}:{}.{} to {}".format(project_id, dataset_id, table_name, destination_uri)
    )


def train_hyper_params(cloudml_client, training_inputs):
    """
    Submit hyper parameters training job to AI platform
    :param cloudml_client: discovery client
    :param training_inputs: spec for the job
    :return: the name of the job created
    """

    job_name = 'chicago_travel_time_training_{}'.format(datetime.utcnow().strftime('%Y%m%d%H%M%S'))
    project_name = 'projects/{}'.format(project_id)
    job_spec = {'jobId': job_name, 'trainingInput': training_inputs}
    response = cloudml_client.projects().jobs().create(body=job_spec,
                                                parent=project_name).execute()
    print(response)

    return job_name


def monitor_training(cloudml_client, job_name):
    """
     Monitors the ongoing training job, when it stops running, returns the results json
    :param cloudml_client:
    :param job_name: the job id to monitor
    :return: job_results
    """

    # wait for job to complete
    job_is_running = True
    while job_is_running:
        job_results = cloudml_client.projects().jobs().get(name='{}/jobs/{}'.format(project_name, job_name)).execute()
        if job_results['state'] in JOB_RUNNING_STATES:

            if 'completedTrialCount' in job_results['trainingOutput']:
                completed_trials = job_results['trainingOutput']['completedTrialCount']
            else:
                completed_trials = 0

            print(str(datetime.utcnow()),
              ': Completed {} training trials'.format(completed_trials),
              ' Waiting for 5 minutes')
            time.sleep(5 * 60)

        elif job_results['state'] in JOB_FAILED_STATES:
            job_is_running = False
            job_results = None

        elif job_results['state'] in JOB_FAILED_STATES:
            job_is_running = False

    return job_results


def create_model(cloudml_client):
    """
    Creates a Model entity in AI Platform
    :param cloudml_client: discovery client
    :return:
    """
    models = cloudml_client.projects().models()
    create_spec = {'name': model_name}

    models.create(body=create_spec,
                  parent=project_name).execute()


def deploy_version(cloudml_client, job_results):
    """
    Deploying the best trail's model to AI platform
    :param cloudml_client: discovery client
    :param job_results: response of the finished AI platform job
    :return:
    """
    models = cloudml_client.projects().models()

    training_outputs = job_results['trainingOutput']
    version_spec = {
        "name": model_version,
        "isDefault": False,
        "runtimeVersion": training_outputs['builtInAlgorithmOutput']['runtimeVersion'],

        # Assuming the trials are sorted by performance (best is first)
        "deploymentUri": training_outputs['trials'][0]['builtInAlgorithmOutput']['modelPath'],
        "framework": training_outputs['builtInAlgorithmOutput']['framework'],
        "pythonVersion": training_outputs['builtInAlgorithmOutput']['pythonVersion'],
        "autoScaling": {
            'minNodes': 0
        }
    }

    versions = models.versions()
    response = versions.create(body=version_spec,
                    parent='{}/models/{}'.format(project_name, model_name)).execute()
    return response


def validate_model():
    """
    Function to validate the model results
    :return:
    """
    df_val = pd.read_csv('{}/processed_data/test.csv'.format(job_dir))
    instances = [", ".join(x) for x in df_val.iloc[:10, 1:].astype(str).values.tolist()]
    service = discovery.build('ml', 'v1')
    version_name = 'projects/{}/models/{}'.format(project_id, model_name)

    if model_version is not None:
        version_name += '/versions/{}'.format(model_version)

    response = service.projects().predict(
        name=version_name,
        body={'instances': instances}
    ).execute()

    if 'error' in response:
        raise RuntimeError(response['error'])

    return response['predictions']



if __name__ == '__main__':
    args = parse_args()
    project_id = args.project_id
    project_name = 'projects/{}'.format(project_id)
    dataset_id = args.dataset_id
    table_name = 'train_{}'.format(datetime.utcnow().strftime('%Y%m%d%H%M%S'))
    model_name = args.model_name
    model_version = args.model_version
    bucket_name = args.bucket_name
    data_dir = "gs://{bucket}/data/{table_name}.csv".format(bucket=bucket_name, table_name=table_name)
    job_dir = "gs://{bucket}/models/{model_version}".format(bucket=bucket_name, model_version=model_version)
    CREATE_MODEL = args.create_model

    with open('hyper_param_spec.dict', 'r')  as f:
        training_inputs = eval(f.read())
        training_inputs['jobDir'] = job_dir
        training_inputs['args'].append('--training_data_path={DATA_DIR}'.format(DATA_DIR=data_dir))

    cloudml_client = discovery.build('ml', 'v1')

    print('Creating training set in BQ...')
    tabel_ref = create_train()

    print('Exporting training set to CGS...')
    export_training_to_gcs(tabel_ref)

    print('Submit hyperparmeter training...')
    job_name = train_hyper_params(cloudml_client, training_inputs)

    print('Waiting for training to finish...')
    job_results = monitor_training(cloudml_client, job_name)

    if CREATE_MODEL == True:
        print('Creating model')
        create_model(cloudml_client)

    print('Deploying version...')
    deploy_version(cloudml_client, job_results)

    print('Validating model...')
    validate_model()
