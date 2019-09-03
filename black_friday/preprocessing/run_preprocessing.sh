CLUSTER_NAME=''
BUCKET=''
TRAIN_FILE="preprocessing.py"
GCS_TRAIN_FILE=gs://$BUCKET/src/
REGION=us-east-1
PY_FILE=$GCS_TRAIN_FILE$TRAIN_FILE

gcloud dataproc clusters create $CLUSTER_NAME
gsutil cp preprocessing.py TRAIN_FILE
gcloud dataproc jobs submit pyspark $PY_FILE \
    --cluster $CLUSTER_NAME --region $REGION

#call this after the job is completed
#gcloud dataproc clusters delete $CLUSTER_NAME

