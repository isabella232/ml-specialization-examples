# https://github.com/GoogleCloudPlatform/cloudml-samples/tree/wenzhel-sklearn/xgboost/iris

BUCKET_ID=doitintl_black_friday
TRAINING_PACKAGE_PATH="/Users/gad/PycharmProjects/ml-specialization-examples/black_friday/cmle/hyperopt/black_friday_hyper_trainer"
MAIN_TRAINER_MODULE="black_friday_hyper_trainer.trainer"
REGION="us-east1"
RUNTIME_VERSION=1.8
PYTHON_VERSION=3.5
SCALE_TIER=BASIC
JOB_DIR=gs://$BUCKET_ID/$JOB_NAME
TRAIN_FILE="data/train_data.csv/part-00000-6a527ad3-ad67-4928-86a7-9d568115b70f-c000.csv"


JOB_NAME="blackfriday_xgboost_hptuning_$(date +"%Y%m%d_%H%M%S")"
JOB_DIR=gs://$BUCKET_ID/$JOB_NAME

gcloud ai-platform jobs submit training $JOB_NAME \
  --job-dir $JOB_DIR \
  --package-path $TRAINING_PACKAGE_PATH \
  --module-name $MAIN_TRAINER_MODULE \
  --region $REGION \
  --runtime-version $RUNTIME_VERSION \
  --python-version $PYTHON_VERSION \
  --scale-tier $SCALE_TIER \
  --config hyperparam.yaml \
   -- \
   --BUCKET_ID $BUCKET_ID \
   --train-file $TRAIN_FILE
