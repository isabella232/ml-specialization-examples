
TRAINER_PACKAGE_PATH="/Users/gad/PycharmProjects/ml-specialization-examples/black_friday/cmle/trainer"
MAIN_TRAINER_MODULE="trainer.train"
BUCKET=doitintl_black_friday
PACKAGE_STAGING_PATH="gs://$BUCKET"
now=$(date +"%Y%m%d_%H%M%S")
JOB_NAME="black_friday_$now"
JOB_DIR="gs://$BUCKET/training_staging/$JOB_NAME"
MODEL_VERSION=61

REGION="us-east1"

gcloud ai-platform jobs submit training $JOB_NAME \
    --staging-bucket $PACKAGE_STAGING_PATH \
    --job-dir $JOB_DIR  \
    --package-path $TRAINER_PACKAGE_PATH \
    --module-name $MAIN_TRAINER_MODULE \
    --region $REGION \
    --runtime-version=1.13\
    --python-version=3.5\
    --\
    --BUCKET $BUCKET\
    --MODEL_VERSION $MODEL_VERSION


