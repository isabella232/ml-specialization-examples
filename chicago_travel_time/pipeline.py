from google.cloud import bigquery
from datetime import datetime
from googleapiclient import discovery
import time
import pandas as pd

project_id = 'gad-playground-212407'
project_name = 'projects/{}'.format(project_id)
dataset_id = 'chicago_taxi'
table_name = 'train_{}'.format(datetime.utcnow().strftime('%Y%m%d%H%M%S'))
model_name = 'travel_time'
model_version = 'v3'
bucket_name = 'doit-chicago-taxi'
data_dir = "gs://doit-chicago-taxi/data/{}.csv".format(table_name)
job_dir = "gs://doit-chicago-taxi/models/{}".format(model_version)
CREATE_MODEL = False


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

    # Start the query, passing in the extra configuration.
    query_job = client.query(
        sql,
        location='US',
        job_config=job_config)

    query_job.result()
    print('Query results loaded to table {}'.format(table_ref.path))
    return table_ref


def export_training_to_gcs(table_ref):
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

    job_name = 'chicago_travel_time_training_{}'.format(datetime.utcnow().strftime('%Y%m%d%H%M%S'))
    project_name = 'projects/{}'.format(project_id)
    job_spec = {'jobId': job_name, 'trainingInput': training_inputs}
    response = cloudml_client.projects().jobs().create(body=job_spec,
                                                parent=project_name).execute()

    print(response)
    return job_name


def monitor_training(cloudml_client, job_name):
    # wait for job to complete
    job_is_running = True
    while job_is_running:
        job_results = cloudml_client.projects().jobs().get(name='{}/jobs/{}'.format(project_name, job_name)).execute()
        job_is_running = job_results['state'] == 'RUNNING'
        if 'completedTrialCount' in job_results['trainingOutput']:
            completed_trials = job_results['trainingOutput']['completedTrialCount']
        else:  completed_trials = 0

        print(str(datetime.utcnow()),
              ': Completed {} training trials'.format(completed_trials),
              ' Waiting for 5 minutes')
        time.sleep(5 * 60)
    return job_results


def create_model(cloudml_client):
    models = cloudml_client.projects().models()
    create_spec = {'name': model_name}

    models.create(body=create_spec,
                  parent=project_name).execute()


def deploy_version(cloudml_client, job_results):
    models = cloudml_client.projects().models()

    training_outputs = job_results['trainingOutput']
    version_spec = {
        "name": model_version,
        "isDefault": False,
        "runtimeVersion": training_outputs['builtInAlgorithmOutput']['runtimeVersion'],
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
    df_val = pd.read_csv('{}/processed_data/validation.csv'.format(job_dir))
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
    with open('hyper_param_spec.json', 'r')  as f:
        training_inputs = eval(f.read())

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
