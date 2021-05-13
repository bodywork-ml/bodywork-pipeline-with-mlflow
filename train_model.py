"""
This module defines what will happen in 'stage-1-train-model':

- download dataset;
- pre-process data into features and labels;
- train machine learning model; and,
- save model to cloud stirage (AWS S3).
"""
from urllib.request import urlopen
from subprocess import CalledProcessError, run
from typing import Tuple

import mlflow
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.base import BaseEstimator
from sklearn.metrics import f1_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from utils import configure_mlflow

MLFLOW_EXPERIMENT = 'iris-classifier'
DATA_URL = ('http://bodywork-ml-pipeline-project.s3.eu-west-2.amazonaws.com'
            '/data/iris_classification_data.csv')


def main() -> None:
    """Main script to be executed."""
    configure_mlflow(MLFLOW_EXPERIMENT)
    data = download_dataset(DATA_URL)
    features, labels = pre_process_data(data)
    train_model(features, labels)


def download_dataset(url: str) -> pd.DataFrame:
    """Get data from cloud object storage."""
    print(f'Downloading training data from {DATA_URL}.')
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


def get_pipeline_git_commit_hash() -> str:
    """The project's git commit to use as a versioning tag."""
    try:
        git_cmd = run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True,
            check=True,
            encoding='utf-8'
        )
    except CalledProcessError:
        raise RuntimeError('Could not get Git commit hash.')
    return git_cmd.stdout


def train_model(features: np.ndarray, labels: np.ndarray) -> None:
    """Train ML model and register with MLflow."""
    random_state = np.random.randint(0, 100)
    run_name = f'pipeline-{get_pipeline_git_commit_hash()}'
    with mlflow.start_run(run_name=run_name) as training_run:
        X_train, X_test, y_train, y_test = train_test_split(
            features,
            labels,
            test_size=0.2,
            stratify=labels,
            random_state=random_state
        )

        print('Training iris decision tree classifier.')
        mlflow.log_param('random_state', random_state)
        iris_tree_classifier = DecisionTreeClassifier(
            class_weight='balanced',
            random_state=random_state
        )
        iris_tree_classifier.fit(X_train, y_train)
        test_data_predictions = iris_tree_classifier.predict(X_test)
        log_model_metrics(y_test, test_data_predictions)

        print('Registering new model with MLflow.')
        model_name = f'{MLFLOW_EXPERIMENT}--sklearn-decision-tree'
        mlflow.sklearn.log_model(
            sk_model=iris_tree_classifier,
            artifact_path=model_name
        )
        new_model_metadata = mlflow.register_model(
            model_uri=f'runs:/{training_run.info.run_id}/{model_name}',
            name=model_name
        )

        print('Transitioning new model to production.')
        mlflow.tracking.MlflowClient().transition_model_version_stage(
            name=model_name,
            version=int(new_model_metadata.version),
            stage='Production'
        )


if __name__ == '__main__':
    main()
