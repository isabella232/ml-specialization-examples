# https://github.com/GoogleCloudPlatform/cloudml-samples/blob/master/census/estimator/trainer

import multiprocessing
import tensorflow as tf

import trainer.constants as constants
import trainer.featurizer as featurizer


def _decode_csv(line):
    row_columns = tf.expand_dims(line, -1)
    columns = tf.decode_csv(
        row_columns, record_defaults=constants.CSV_COLUMN_DEFAULTS)
    features = dict(zip(constants.CSV_COLUMNS, columns))

    # Remove unused columns
    unused_columns = set(constants.CSV_COLUMNS) - \
        {col.name for col in featurizer.INPUT_COLUMNS} - \
        {constants.LABEL_COLUMN}
    for col in unused_columns:
        features.pop(col)
    return features


def input_fn(filenames,
             num_epochs=None,
             shuffle=True,
             skip_header_lines=0,
             batch_size=200,
             num_parallel_calls=None,
             prefetch_buffer_size=None):

    if num_parallel_calls is None:
        num_parallel_calls = multiprocessing.cpu_count()

    if prefetch_buffer_size is None:
        prefetch_buffer_size = 1024

    dataset = tf.data.TextLineDataset(filenames).skip(skip_header_lines).map(
        _decode_csv, num_parallel_calls).prefetch(prefetch_buffer_size)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=batch_size * 10)
    iterator = dataset.repeat(num_epochs).batch(
        batch_size).make_one_shot_iterator()
    features = iterator.get_next()
    return features, features.pop(constants.LABEL_COLUMN)


def csv_serving_input_fn():
    """Build the serving inputs."""
    csv_row = tf.placeholder(shape=[None], dtype=tf.string)
    features = _decode_csv(csv_row)
    features.pop(constants.LABEL_COLUMN)
    return tf.estimator.export.ServingInputReceiver(features,
                                                    {'csv_row': csv_row})


def example_serving_input_fn():
    """Build the serving inputs."""
    example_bytestring = tf.placeholder(
        shape=[None],
        dtype=tf.string,
    )
    features = tf.parse_example(
        example_bytestring,
        tf.feature_column.make_parse_example_spec(featurizer.INPUT_COLUMNS))
    return tf.estimator.export.ServingInputReceiver(
        features, {'example_proto': example_bytestring})


def json_serving_input_fn():
    """Build the serving inputs."""
    inputs = {}
    for feat in featurizer.INPUT_COLUMNS:
        inputs[feat.name] = tf.placeholder(shape=[None], dtype=feat.dtype)

    return tf.estimator.export.ServingInputReceiver(inputs, inputs)


SERVING_FUNCTIONS = {
    'JSON': json_serving_input_fn,
    'EXAMPLE': example_serving_input_fn,
    'CSV': csv_serving_input_fn
}
