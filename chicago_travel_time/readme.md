### Chicago Trip Time Model

- Machine Learning pipeline to predict travel time of Chicago taxi rides based on pickup and dropoff locations and the time of ride start.
- The pipeline collects a sample data from BigQuery open dataset, preprocesses, launches an AI platform hyper parameters training job and deploys the model.
- The model built is a Tensorflow deep neural network.

## Running the model
#### Parameters
- **project_id** - ID of the Google Cloud project in which you run. must have at least Editor role
- **dataset_id** - Create a BigQuery Dataset to store training datasets
- **model_name** - Name of the model that will be used in AI platform
- **model_version** - Name of the model version that will be used in AI platform 
- **bucket_name** - Create a Google Cloud Storage bucket to store training artifacts 
- **create_model** - Boolean, should the pipeline create a new AI platrform model?

#### Submit pipeline job
```bash
python3 pipeline.py --project_id PROJECT_ID   --dataset_id DATASET_ID --model_name MODEL_NAME --model_version MODEL_VERSION --bucket_name BUCKET --create_model [True|False]
```