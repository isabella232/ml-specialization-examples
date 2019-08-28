CLUSTER_NAME=''
gcloud dataproc clusters create $CLUSTER_NAME
gcloud dataproc jobs submit job-command \
    --cluster $CLUSTER_NAME --region region

