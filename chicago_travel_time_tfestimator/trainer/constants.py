# https://github.com/GoogleCloudPlatform/cloudml-samples/blob/master/census/estimator/trainer
"""Constants shared by preprocessing and modeling scripts."""

# Define the format of your input data as present in the CSV file

CSV_COLUMNS = [
 'trip_seconds',
 'air_distance',
 'pickup_latitude',
 'pickup_longitude',
 'dropoff_latitude',
 'dropoff_longitude',
 'pickup_community_area',
 'dropoff_community_area',
 'trip_start_hour',
 'trip_start_weekday',
 'trip_start_week',
 'trip_start_yearday',
 'trip_start_month'
]
CSV_COLUMN_DEFAULTS = [[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0], [0], [0], [0], [0], [0], [0]]

LABEL_COLUMN = 'trip_seconds'