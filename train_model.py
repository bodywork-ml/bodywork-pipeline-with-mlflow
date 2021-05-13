"""
This module defines what will happen in 'stage-1-train-model':

- download dataset;
- pre-process data into features and labels;
- train machine learning model; and,
- save model to cloud stirage (AWS S3).
"""
import os
from urllib.request import urlopen
from typing import Tuple

import mlflow
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.base import BaseEstimator
from sklearn.metrics import f1_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

MLFLOW_EXPERIMENT = 'iris-classifier'
DATA_URL = ('http://bodywork-ml-pipeline-project.s3.eu-west-2.amazonaws.com'
            '/data/iris_classification_data.csv')


def main() -> None:
    """Main script to be executed."""
    configure_mlflow()
    data = download_dataset(DATA_URL)
    features, labels = pre_process_data(data)
    train_model(features, labels)


def configure_mlflow() -> None:
    """Set tracking server URL and experiment.

    The MLflow client requires that you pass it the URL of the MLflow
    tracking server and that the client has the appropriate credentials
    to access the object storage for logging models (which in this case
    is via the AWS client library that is used to access object storage
    provided by Minio).
    """
    try:
        mlflow_tracking_uri = os.environ['MLFLOW_TRACKING_URI']
    except KeyError:
        raise RuntimeError('cannot find env var MLFLOW_TRACKING_URI')

    try:
        os.environ['AWS_ACCESS_KEY_ID']
        os.environ['AWS_SECRET_ACCESS_KEY']
    except KeyError:
        msg = 'cannot find env var AWS_ACCESS_KEY_ID or AWS_SECRET_ACCESS_KEY'
        raise RuntimeError(msg)

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)


def download_dataset(url: str) -> pd.DataFrame:
    """Get data from cloud object storage."""
    print(f'downloading training data from {DATA_URL}')
    data_file = urlopen(url)
    return pd.read_csv(data_file)


def pre_process_data(data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare raw data for model training."""
    label_column = 'species'
    feature_columns = [
        'sepal length (cm)',
        'sepal width (cm)',
        'petal length (cm)',
        'petal width (cm)'
    ]
    classes_map = {'setosa': 0, 'versicolor': 1, 'virginica': 2}
    X = data[feature_columns].values
    y = data[label_column].apply(lambda e: classes_map[e]).values
    return X, y


def log_model_metrics(
    y_actual: np.ndarray,
    y_predicted: np.ndarray
) -> None:
    """Print model evaluation metrics to stdout."""
    accuracy = balanced_accuracy_score(
        y_actual,
        y_predicted,
        adjusted=True
    )
    f1 = f1_score(
        y_actual,
        y_predicted,
        average='weighted'
    )
    mlflow.log_metric('accuracy', accuracy)
    mlflow.log_metric('f1', f1)


def train_model(features: np.ndarray, labels: np.ndarray) -> None:
    """Train ML model and register with MLflow."""
    with mlflow.start_run(run_name='retraining') as training_run:
        random_state = np.random.randint(0, 100)
        mlflow.log_param('random_state', random_state)

        X_train, X_test, y_train, y_test = train_test_split(
            features,
            labels,
            test_size=0.1,
            stratify=labels,
            random_state=random_state
        )

        print('training iris decision tree classifier')
        iris_tree_classifier = DecisionTreeClassifier(
            class_weight='balanced',
            random_state=random_state
        )
        iris_tree_classifier.fit(X_train, y_train)
        test_data_predictions = iris_tree_classifier.predict(X_test)
        log_model_metrics(y_test, test_data_predictions)

        print('registering new model with MLflow')
        model_name = 'sklearn-decision-tree-classifier'
        mlflow.sklearn.log_model(
            sk_model=iris_tree_classifier,
            artifact_path=model_name
        )
        new_model_metadata = mlflow.register_model(
            model_uri=f'runs:/{training_run.info.run_id}/{model_name}',
            name=model_name
        )

        print('transitioning new model to production')
        mlflow.tracking.MlflowClient().transition_model_version_stage(
            name=model_name,
            version=int(new_model_metadata.version),
            stage='Production'
        )


if __name__ == '__main__':
    main()
