# https://github.com/GoogleCloudPlatform/cloudml-samples/tree/wenzhel-sklearn/xgboost/iris

PROJECT_ID=gad-playground-212407
BUCKET_ID=doitintl_black_friday
TRAINING_PACKAGE_PATH="[YOUR-LOCAL-PATH-TO-TRAINING-PACKAGE]/iris_xgboost_trainer/"
MAIN_TRAINER_MODULE="iris_xgboost_trainer.iris"
REGION=[REGION]
RUNTIME_VERSION=1.8
PYTHON_VERSION=2.7
SCALE_TIER=BASIC
JOB_DIR=gs://$BUCKET_ID/$JOB_NAME

JOB_NAME="blackfriday_xgboost_hptuning_$(date +"%Y%m%d_%H%M%S")"
JOB_DIR=gs://$BUCKET_ID/$JOB_NAME
gcloud ml-engine jobs submit training $JOB_NAME \
  --job-dir $JOB_DIR \
  --package-path $TRAINING_PACKAGE_PATH \
  --module-name $MAIN_TRAINER_MODULE \
  --region us-central1 \
  --runtime-version $RUNTIME_VERSION \
  --python-version $PYTHON_VERSION \
  --scale-tier BASIC \
  --config hyperparam.yaml