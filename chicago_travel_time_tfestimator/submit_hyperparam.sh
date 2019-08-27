# https://github.com/GoogleCloudPlatform/cloudml-samples/tree/wenzhel-sklearn/xgboost/iris

BUCKET=doit-chicago-taxi
PROJECT_ID=gad-playground-212407
DATASET_ID=chicago_taxi
TRAINER_PACKAGE_PATH="/Users/gad/PycharmProjects/ml-specialization-examples/chicago_travel_time_tfestimator/trainer"
MAIN_TRAINER_MODULE="trainer.task"
TIER=BASIC_GPU
REGION="us-east1"
RUNTIME_VERSION=1.13
PYTHON_VERSION=3.5
SCALE_TIER=BASIC
JOB_NAME="chicago_hptuning_$(date +"%Y%m%d_%H%M%S")"
JOB_DIR=gs://$BUCKET/$JOB_NAME
PACKAGE_STAGING_PATH="gs://$BUCKET"
TRAIN_DATA="gs://doit-chicago-taxi/data/train_20190826230154.csv"
VAL_DATA="gs://doit-chicago-taxi/data/val_20190826230154.csv"

gcloud ai-platform jobs submit training $JOB_NAME \
  --job-dir $JOB_DIR \
  --staging-bucket $PACKAGE_STAGING_PATH \
  --package-path $TRAINER_PACKAGE_PATH \
  --module-name $MAIN_TRAINER_MODULE \
  --region $REGION \
  --runtime-version $RUNTIME_VERSION \
  --python-version $PYTHON_VERSION \
  --scale-tier $SCALE_TIER \
  --config hyperparam.yaml \
   -- \
  --num-epochs 10 \
  --CREATE_DATASET "FALSE" \
  --train-data $TRAIN_DATA \
  --val-data $VAL_DATA