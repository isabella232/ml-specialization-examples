# Black Friday Purchase Prediction Demo

[![N|Solid](https://cdn-images-1.medium.com/max/100/1*IRQJJ9A7YiUQzYrGToppFw@2x.png)](https://doit-intl.com)

Demo machine learning pipeline on Google Cloud

  - Training and Serving XGBoost using AI platform
  - Web Service hosted on App Engine

# How to setup
1. Export the a dataset to GCS and train the model
```sh
$ bash ml-specialization-examples/black_friday/cmle/submit_training.sh
```

2. deploy the model to cloud ml using the UI

3. deploy the web app using
```sh
$ gcloud app deploy app.yaml 
```

___
## model training
The code runs with default xgboost params. You can run a hyper parameters tuning job using AI platform with the hyper parameter tuning script
```sh
$ bash submit_hyperparam.sh
```
