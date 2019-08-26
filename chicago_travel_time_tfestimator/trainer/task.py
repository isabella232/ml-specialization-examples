# https://cloud.google.com/ml-engine/docs/using-hyperparameter-tuning
# https://github.com/GoogleCloudPlatform/cloudml-samples/blob/master/census/estimator/trainer/task.py


import argparse
import json
import os
import tensorflow as tf
import trainer.input_funcs as input_module
import trainer.model as model
from trainer.dataset import create_datasets


def _get_session_config_from_env_var():
    """Returns a tf.ConfigProto instance that has appropriate device_filters
    set."""

    tf_config = json.loads(os.environ.get('TF_CONFIG', '{}'))

    if (tf_config and 'task' in tf_config and 'type' in tf_config['task'] and
            'index' in tf_config['task']):
        # Master should only communicate with itself and ps
        if tf_config['task']['type'] == 'master':
            return tf.ConfigProto(device_filters=['/job:ps', '/job:master'])
        # Worker should only communicate with itself and ps
        elif tf_config['task']['type'] == 'worker':
            return tf.ConfigProto(device_filters=[
                '/job:ps',
                '/job:worker/task:%d' % tf_config['task']['index']
            ])
    return None

def metric(labels, predictions):
    pred_values = predictions['predictions']
    return {'rmse': tf.metrics.root_mean_squared_error(labels, pred_values),
            'mae': tf.metrics.mean_absolute_error(labels, pred_values),
            'mean_rel_error': tf.metrics.mean_relative_error(labels, pred_values, labels)}

def train_and_evaluate(args):
    """Run the training and evaluate using the high level API."""

    def train_input():
        """Input function returning batches from the training
        data set from training.
        """
        return input_module.input_fn(
            args.train_files,
            num_epochs=args.num_epochs,
            batch_size=args.train_batch_size,
            num_parallel_calls=args.num_parallel_calls,
            prefetch_buffer_size=args.prefetch_buffer_size)

    def eval_input():
        """Input function returning the entire validation data
        set for evaluation. Shuffling is not required.
        """
        return input_module.input_fn(
            args.eval_files,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_parallel_calls=args.num_parallel_calls,
            prefetch_buffer_size=args.prefetch_buffer_size)

    train_spec = tf.estimator.TrainSpec(
        train_input, max_steps=args.train_steps)

    exporter = tf.estimator.FinalExporter(
        'chicago-taxi', input_module.SERVING_FUNCTIONS[args.export_format])

    eval_spec = tf.estimator.EvalSpec(
        eval_input,
        steps=args.eval_steps,
        exporters=[exporter],
        name='chicago-eval')

    run_config = tf.estimator.RunConfig(
        session_config=_get_session_config_from_env_var())
    run_config = run_config.replace(model_dir=args.job_dir)
    print('Model dir %s' % run_config.model_dir)
    estimator = model.build_estimator(
        embedding_size=args.embedding_size,
        # Construct layers sizes with exponential decay
        hidden_units=[
            max(2, int(args.first_layer_size * args.scale_factor**i))
            for i in range(args.num_layers)
        ],
        config=run_config)

    estimator = tf.contrib.estimator.add_metrics(estimator, metric)

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser()
    # Input Arguments
    PARSER.add_argument(
        '--train-files',
        help='GCS file or local paths to training data',
        nargs='+',
        default='gs://doit-chicago-taxi/data/train_20190606101022.csv')
    PARSER.add_argument(
        '--eval-files',
        help='GCS file or local paths to evaluation data',
        nargs='+',
        default='gs://doit-chicago-taxi/data/train_20190606100533.csv')
    PARSER.add_argument(
        '--job-dir',
        help='GCS location to write checkpoints and export models',
        default='/tmp/tensorboard-logs/model-1')
    PARSER.add_argument(
        '--num-parallel-calls',
        help='Number of threads used to read in parallel the training and evaluation',
        type=int)
    PARSER.add_argument(
        '--prefetch_buffer_size',
        help='Naximum number of input elements that will be buffered when prefetching',
        type=int)
    PARSER.add_argument(
        '--num-epochs',
        help="""\
      Maximum number of training data epochs on which to train.
      If both --max-steps and --num-epochs are specified,
      the training job will run for --max-steps or --num-epochs,
      whichever occurs first. If unspecified will run for --max-steps.\
      """,
        type=int,
        default=10)
    PARSER.add_argument(
        '--train-batch-size',
        help='Batch size for training steps',
        type=int,
        default=1024)
    PARSER.add_argument(
        '--eval-batch-size',
        help='Batch size for evaluation steps',
        type=int,
        default=256)
    PARSER.add_argument(
        '--embedding-size',
        help='Number of embedding dimensions for categorical columns',
        default=8,
        type=int)
    PARSER.add_argument(
        '--first-layer-size',
        help='Number of nodes in the first layer of the DNN',
        default=100,
        type=int)
    PARSER.add_argument(
        '--num-layers',
        help='Number of layers in the DNN',
        default=4,
        type=int)
    PARSER.add_argument(
        '--scale-factor',
        help='How quickly should the size of the layers in the DNN decay',
        default=0.7,
        type=float)
    PARSER.add_argument(
        '--train-steps',
        help="""\
      Steps to run the training job for. If --num-epochs is not specified,
      this must be. Otherwise the training job will run indefinitely.""",
        type=int)
    PARSER.add_argument(
        '--eval-steps',
        help='Number of steps to run evalution for at each checkpoint',
        default=100,
        type=int)
    PARSER.add_argument(
        '--export-format',
        help='The input format of the exported SavedModel binary',
        choices=['JSON', 'CSV', 'EXAMPLE'],
        default='JSON')
    PARSER.add_argument(
        '--verbosity',
        choices=['DEBUG', 'ERROR', 'FATAL', 'INFO', 'WARN'],
        default='INFO')
    PARSER.add_argument(
        '--BUCKET',
        help='bucket to store the data for the training')
    PARSER.add_argument(
        '--PROJECT_ID',
        help='Google Cloud project id in which you run')
    PARSER.add_argument(
        '--dataset_id',
        help='BigQuery Dataset ID in which you store the datasets')

    ARGUMENTS, _ = PARSER.parse_known_args()

    # Set python level verbosity
    tf.compat.v1.logging.set_verbosity(ARGUMENTS.verbosity)
    # Set C++ Graph Execution level verbosity
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(
        tf.compat.v1.logging.__dict__[ARGUMENTS.verbosity] / 10)

    dadaset_paths = create_datasets(ARGUMENTS.BUCKET, ARGUMENTS.project_id, ARGUMENTS.dataset_id)
    ARGUMENTS['train_files'] = dadaset_paths['train_path']
    ARGUMENTS['eval_files'] = dadaset_paths['val_path_path']

    # Run the training job
    train_and_evaluate(ARGUMENTS)



