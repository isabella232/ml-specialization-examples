# https://github.com/GoogleCloudPlatform/cloudml-samples/blob/master/census/estimator/trainer

"""Defines a Wide + Deep model for classification on structured data.

Tutorial on wide and deep: https://www.tensorflow.org/tutorials/wide_and_deep/
"""

import tensorflow as tf


# Define the initial ingestion of each feature used by your model.
# Additionally, provide metadata about the feature.
INPUT_COLUMNS = [
    # Categorical base columns

    # For categorical columns with known values we can provide lists
    # of values ahead of time.
    tf.feature_column.categorical_column_with_vocabulary_list(
        'trip_start_hour', list(range(24))),
    
    tf.feature_column.categorical_column_with_vocabulary_list(
        'trip_start_weekday', list(range(7))),
    
    tf.feature_column.categorical_column_with_vocabulary_list(
        'trip_start_week', list(range(53))
    ),
    tf.feature_column.categorical_column_with_vocabulary_list(
        'trip_start_month', list(range(1, 13))),
    
    # For columns with a large number of values, or unknown values
    # We can use a hash function to convert to categories.
    tf.feature_column.categorical_column_with_hash_bucket(
        'pickup_community_area', hash_bucket_size=32, dtype=tf.int32),
    tf.feature_column.categorical_column_with_hash_bucket(
        'dropoff_community_area', hash_bucket_size=32, dtype=tf.int32),
    tf.feature_column.categorical_column_with_hash_bucket(
        'trip_start_yearday', hash_bucket_size=128, dtype=tf.int32),
    
    # Continuous base columns.
    tf.feature_column.numeric_column('air_distance'),
    tf.feature_column.numeric_column('pickup_latitude'),
    tf.feature_column.numeric_column('pickup_longitude'),
    tf.feature_column.numeric_column('dropoff_latitude'),
    tf.feature_column.numeric_column('dropoff_longitude'),
]



def get_deep_and_wide_columns(embedding_size=8):
    """Creates deep and wide feature_column lists.
    Args:
            embedding_size: (int), the number of dimensions used to represent categorical
                                   features when providing them as inputs to the DNN.
    Returns:
            [tf.feature_column],[tf.feature_column]: deep and wide feature_column lists.
    """

    (trip_start_hour, trip_start_weekday, trip_start_week,
     trip_start_month, pickup_community_area,
     dropoff_community_area, trip_start_yearday,
     air_distance, pickup_latitude, pickup_longitude,
     dropoff_latitude, dropoff_longitude) = INPUT_COLUMNS

    # Wide columns and deep columns.
    wide_columns = [
        # Interactions between different categorical features can also
        # be added as new virtual features.
        tf.feature_column.crossed_column(['trip_start_hour', 'trip_start_weekday'],
                                         hash_bucket_size=int(1e4)),
        tf.feature_column.crossed_column(['trip_start_hour', 'trip_start_yearday'],
                                         hash_bucket_size=int(1e4)),
        trip_start_hour,
        trip_start_weekday,
        trip_start_week,
        trip_start_month
    ]

    deep_columns = [
        # Use indicator columns for low dimensional vocabularies
        tf.feature_column.indicator_column(trip_start_hour),
        tf.feature_column.indicator_column(trip_start_weekday),
        tf.feature_column.indicator_column(trip_start_week),
        tf.feature_column.indicator_column(trip_start_month),

        # Use embedding columns for high dimensional vocabularies
        tf.feature_column.embedding_column(
            pickup_community_area, dimension=embedding_size),
        tf.feature_column.embedding_column(
            dropoff_community_area, dimension=embedding_size),
        tf.feature_column.embedding_column(
            trip_start_yearday, dimension=embedding_size),
   
        air_distance, 
        pickup_latitude,
        pickup_longitude,
        dropoff_latitude,
        dropoff_longitude,
    ]

    return deep_columns, wide_columns
