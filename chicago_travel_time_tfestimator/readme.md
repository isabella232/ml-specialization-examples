### Chicago Trip Time Model

- Machine Learning pipeline to predict travel time of Chicago taxi rides based on pickup and dropoff locations and the time of ride start.
- The pipeline collects a sample data from BigQuery open dataset, preprocesses, launches an AI platform hyper parameters training job and trains the final.
- The model built is a Tensorflow wide and deep neural network.

## Creating a dataset
Before running a hyper parameter training job it is required that you create a training and validation set. 
This can be done by calling the dataset creation code as follows:

#### Parameters
- **project** - ID of the Google Cloud project in which you run. must have at least Editor role
- **dataset-id** - Create a BigQuery Dataset to store training datasets,
                   this is the name of the dataset without project prefix
- **bucket-name** - Create a Google Cloud Storage bucket to store training artifacts 

#### Creating the datasets
```bash
bash ./trainer/dataset.py --project-id PROJECT_ID \
                    --dataset-id DATASET_IID\
                    --bucket-name BUCKET\
```

## Submitting hyper-parameter tuning job
In order to find optimal training parameters for the model, we recommend to run the hyperparameter training job.
You will need to modify some parameters in the submit_hyperparam.sh file. the rest of the params will be set automatically.
#### Parameters
- **BUCKET** - the staging bucket for the AI-Platform job
- **PROJECT_ID** - the project id in which you run
- **DATASET_ID** - Create a BigQuery Dataset to store training datasets,
                    this is the name of the dataset without project prefix
- **TRAINER_PACKAGE_PATH** - path to the training code on the machine from which you run the training command
- **TRAIN_DATA** - path to the training csv created in the previous stage
- **VAL_DATA** - path to the validation csv created in the previous stage

#### Submit pipeline job
```bash
bash submit_hyperparam.sh

```

## Submitting final model training
After discovering the optimal parameters you can either deploy the optimal model you trained in the hyper parameter tuning job, or train your own model.
The training job will create a new dataset (by default). You will need to change the hyper-parameters to the optimal parameters you discovered.
You will need to modify some parameters in the submit_training.sh file. the rest of the params will be set automatically.
 
#### Parameters
- **BUCKET** - the staging bucket for the AI-Platform job
- **PROJECT_ID** - the project id in which you run
- **DATASET_ID** - Create a BigQuery Dataset to store training datasets,
                    this is the name of the dataset without project prefix
- **CREATE_DATASET** - FALSE or TRUE based on whether you wish to create a training file or provide one.

#### Submit pipeline job
```bash
bash submit_training.sh

```
