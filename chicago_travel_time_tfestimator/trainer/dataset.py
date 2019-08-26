from google.cloud import bigquery
from datetime import datetime

DATASET_QUERY = """
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
                    AND  {WHERE_CLAUSE}
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

def execute_query(query, table_name, dataset_id):
    client = bigquery.Client()
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


def create_train(train_table_name, dataset_id):
    query = DATASET_QUERY.format(WHERE_CLAUSE="trip_start_timestamp < '2016-01-01'")
    return execute_query(query, train_table_name, dataset_id)


def create_validation(validation_table_name, dataset_id):
    query = DATASET_QUERY.format(WHERE_CLAUSE="trip_start_timestamp >= '2016-01-01'")
    return execute_query(query, validation_table_name, dataset_id)


def export_table_to_gcs(table_ref, table_name, data_dir, project_id, dataset_id):
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


def create_datasets(bucket_name, project_id, dataset_id):
    now = datetime.utcnow().strftime('%Y%m%d%H%M%S')
    train_table_name = 'train_{}'.format(now)
    validation_table_name = 'val_{}'.format(now)
    data_dir = "gs://{bucket}/data/{table_name}.csv".format(bucket=bucket_name, table_name=train_table_name)
    val_dir = "gs://{bucket}/data/{table_name}.csv".format(bucket=bucket_name, table_name=validation_table_name)

    print('Creating training set in BQ...')
    tabel_ref = create_train(train_table_name, dataset_id)

    print('Exporting training set to CGS...')
    export_table_to_gcs(tabel_ref, train_table_name,  data_dir, project_id, dataset_id)

    print('Creating validation set in BQ...')
    tabel_ref = create_validation(validation_table_name, dataset_id)

    print('Exporting validation set to CGS...')
    export_table_to_gcs(tabel_ref, validation_table_name, val_dir, project_id, dataset_id)

    return {'train_path': data_dir,
            'val_path': val_dir}
