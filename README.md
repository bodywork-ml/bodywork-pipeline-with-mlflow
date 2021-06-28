# A Train-and-Serve Pipeline using MLflow for Tracking Metrics and Managing Models

![bodywork](https://bodywork-media.s3.eu-west-2.amazonaws.com/ml_pipeline_with_mlflow.png)

This repository contains a project that demonstrates how to use Bodywork to deploy a train-and-serve pipeline on Kubernetes. The pipeline uses MLflow to track the training metrics and manage the persistence of trained models. The pipeline consists of two stages:

1. Run a batch job to train a model, logging metrics and registering models to MLflow.
2. Deploy a scoring service with a REST API, that loads the latest 'production' model from MLflow.

## Getting Access to a Kubernetes Cluster

In order to run this example project you will need access to a Kubernetes cluster. If you need help help with this, then see our [introductory guide to Kubernetes for ML](https://bodywork.readthedocs.io/en/latest/kubernetes/#getting-started-with-kubernetes) - we'll have you setup with a local test cluster in 10 minutes.

## Deploying MLflow

You will need to have access to the MLflow tracking server. Should you need to, you can also use Bodywork to deploy MLflow to your cluster - see [here](https://www.bodyworkml.com/posts/deploy-mlflow-with-bodywork) for a complete guide to deploying a production-grade instance of MLflow, to Kubernetes. When you're ready, check your access to Kubernetes by running,

```text
$ kubectl cluster-info
```

## Deploying the Pipeline

To run this project, follow the steps below.

### Step 1 - Install the Bodywork Python Package

```text
$ pip install bodywork
```

### Step 2 - Setup a Kubernetes Namespace for use with Bodywork

```text
$ bodywork setup-namespace bodywork-mlflow-demo
```

### Step 3 - Inject Credentials for MLflow

The MLflow client requires the appropriate credentials for reading/writing to the file storage system that you have chosen to use as your MLflow artefact repository. For example, we use AWS S3 to support the MLflow artefact repository, which means that MLflow will automatically use the AWS client library (Boto3) for accessing S3. This requires specific environment variables to be set using the required AWS credentials. These can be injected into Bodywork containers by first of all creating a Kubernetes secret to hold the necessary data,

```text
bodywork secret create \
    --namespace=bodywork-mlflow-demo \
    --name=mlflow-credentials \
    --data AWS_ACCESS_KEY_ID=X \
           AWS_SECRET_ACCESS_KEY=X \ 
           AWS_DEFAULT_REGION=X \
           MLFLOW_TRACKING_URI=http://bodywork-mlflow--server.mlflow.svc.cluster.local:5000 \
           MLFLOW_S3_ENDPOINT_URL=null
```

And then configuring each stage to look for the secrets, in the `bodywork.yaml` configuration file. For example,

```yaml
stages:
  train_model:
    ...
    secrets:
      AWS_ACCESS_KEY_ID: mlflow-credentials
      AWS_SECRET_ACCESS_KEY: mlflow-credentials
      AWS_DEFAULT_REGION: mlflow-credentials
      MLFLOW_TRACKING_URI: mlflow-credentials
      MLFLOW_S3_ENDPOINT_URL: mlflow-credentials
```

Note, that we have used the same secret for configuring the MLflow tracking URI and S3 endpoint URL. For the former, we use the domain name assigned to the MLflow tracking server within the cluster, while the latter is optional (set to `null` is not required), and made available for when you want to use other S3 compatible storage types (e.g. [Minio](https://min.io)).

### Step 4 - Deploy the Pipeline

To trigger the deployment run,

```text
$ bodywork deployment create \
    --namespace=bodywork-mlflow-demo \
    --name=initial-deployment \
    --git-repo-url=https://github.com/bodywork-ml/bodywork-pipeline-with-mlflow \
    --git-repo-branch=master \
    --local-workflow-controller
```

You can check on the deployment's progress by using,

```text
$ bodywork deployment display \
    --namespace=bodywork-mlflow-demo \
    --name=initial-deployment
```

Once the deployment has completed, browse to the MLflow UI to check on the model metrics that were logged during training.

## Testing the Model-Scoring Service

Service deployments are accessible via HTTP from within the cluster - they are not exposed to the public internet, unless you have [installed an ingress controller](https://bodywork.readthedocs.io/en/latest/kubernetes/#configuring-ingress) in your cluster. The simplest way to test a service from your local machine, is by using a local proxy server to enable access to your cluster. This can be achieved by issuing the following command,

```text
$ kubectl proxy
```

Then in a new shell, you can use the curl tool to test the service. For example,

```text
$ curl http://localhost:8001/api/v1/namespaces/bodywork-mlflow-demo/services/bodywork-mlflow-demo--scoring-service/proxy/iris/v1/score \
    --request POST \
    --header "Content-Type: application/json" \
    --data '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
```

Should return,

```json
{
    "species_prediction":"setosa",
    "probabilities":"setosa=1.0|versicolor=0.0|virginica=0.0",
    "model_info": "DecisionTreeClassifier(class_weight='balanced', random_state=42)"
}
```

According to how the payload has been defined in the `serve_model.py` module.

If an ingress controller is operational in your cluster, then the service can be tested via the public internet using,

```text
$ curl http://YOUR_CLUSTERS_EXTERNAL_IP/bodywork-mlflow-demo/bodywork-mlflow-demo--scoring-service/iris/v1/score \
    --request POST \
    --header "Content-Type: application/json" \
    --data '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
```

See [here](https://bodywork.readthedocs.io/en/latest/kubernetes/#connecting-to-the-cluster) for instruction on how to retrieve `YOUR_CLUSTERS_EXTERNAL_IP`.

## Running the ML Pipeline on a Schedule

If you're happy with the test results, you can schedule the workflow-controller to operate remotely on the cluster, on a pre-defined schedule. For example, to setup the the workflow to run every hour, use the following command,

```text
$ bodywork cronjob create \
    --namespace=bodywork-mlflow-demo \
    --name=train-and-deploy \
    --schedule="0 * * * *" \
    --git-repo-url=https://github.com/bodywork-ml/bodywork-bodywork-mlflow-demo-project \
    --git-repo-branch=master
```

Each scheduled workflow will attempt to re-run the batch-job, as defined by the state of this repository's `master` branch at the time of execution.

To get the execution history for all `train-and-deploy` jobs use,

```text
$ bodywork cronjob history \
    --namespace=bodywork-mlflow-demo \
    --name=train-and-deploy
```

Which should return output along the lines of,

```text
JOB_NAME                                START_TIME                    COMPLETION_TIME               ACTIVE      SUCCEEDED       FAILED
train-and-deploy-1605214260             2020-11-12 20:51:04+00:00     2020-11-12 20:52:34+00:00     0           1               0
```

Then to stream the logs from any given cronjob run (e.g. to debug and/or monitor for errors), use,

```text
$ bodywork cronjob logs \
    --namespace=bodywork-mlflow-demo \
    --name=train-and-deploy-1605214260
```

## Cleaning Up

To clean-up the deployment in its entirety, delete the namespace using kubectl - e.g. by running,

```text
$ kubectl delete ns bodywork-mlflow-demo
```

## Make this Project Your Own

This repository is a [GitHub template repository](https://docs.github.com/en/free-pro-team@latest/github/creating-cloning-and-archiving-repositories/creating-a-repository-from-a-template) that can be automatically copied into your own GitHub account by clicking the `Use this template` button above.

After you've cloned the template project, use official [Bodywork documentation](https://bodywork.readthedocs.io/en/latest/) to help modify the project to meet your own requirements.
