
TRAINER_PACKAGE_PATH="[LOCAL_BASE_DIR]/ml-specialization-examples/chicago_travel_time_tfestimator/trainer"
MAIN_TRAINER_MODULE="trainer.task"
BUCKET=[BUCKET]
PROJECT_ID=[PROJECT_ID]
DATASET_ID=chicago_taxi
PACKAGE_STAGING_PATH="gs://$BUCKET"
now=$(date +"%Y%m%d_%H%M%S")
JOB_NAME="chicago_travel_time_tfestimator_$now"
JOB_DIR="gs://$BUCKET/training_staging/$JOB_NAME"
TIER=BASIC_GPU
REGION="us-east1"

gcloud ai-platform jobs submit training $JOB_NAME \
    --staging-bucket $PACKAGE_STAGING_PATH \
    --job-dir $JOB_DIR  \
    --package-path $TRAINER_PACKAGE_PATH \
    --module-name $MAIN_TRAINER_MODULE \
    --region $REGION \
    --runtime-version=1.13\
    --python-version=3.5 \
    --scale-tier $TIER \
    -- \
    --CREATE_DATASET "TRUE" \
    --BUCKET $BUCKET \
    --PROJECT_ID $PROJECT_ID \
    --DATASET_ID $DATASET_ID
