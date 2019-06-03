# https://github.com/GoogleCloudPlatform/cloudml-samples/tree/wenzhel-sklearn/xgboost/iris

PROJECT_ID=<YOUR_PROJECT>
BUCKET_ID=doitintl_black_friday
TRAINING_PACKAGE_PATH="<LOCAL TRAINING PATH>/black_friday_hyper_trainer/"
MAIN_TRAINER_MODULE="black_friday_hyper_trainer.trainer"
REGION=<REGION>
RUNTIME_VERSION=1.8
PYTHON_VERSION=3.5
SCALE_TIER=BASIC
JOB_DIR=gs://$BUCKET_ID/$JOB_NAME

JOB_NAME="blackfriday_xgboost_hptuning_$(date +"%Y%m%d_%H%M%S")"
JOB_DIR=gs://$BUCKET_ID/$JOB_NAME
gcloud ml-engine jobs submit training $JOB_NAME \
  --job-dir $JOB_DIR \
  --package-path $TRAINING_PACKAGE_PATH \
  --module-name $MAIN_TRAINER_MODULE \
  --region #$REGION \
  --runtime-version $RUNTIME_VERSION \
  --python-version $PYTHON_VERSION \
  --scale-tier BASIC \
  --config hyperparam.yaml \
   -- \
   --BUCKET $BUCKET_ID