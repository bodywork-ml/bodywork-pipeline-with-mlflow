"""
This module contains helper functions used in multiple stages.
"""
import logging
import os
import sys

import mlflow


def configure_logger() -> logging.Logger:
    """Configure a logger that will write to stdout."""
    log_handler = logging.StreamHandler(sys.stdout)
    log_format = logging.Formatter(
        "%(asctime)s - " "%(levelname)s - " "%(module)s.%(funcName)s - " "%(message)s"
    )
    log_handler.setFormatter(log_format)
    log = logging.getLogger(__name__)
    log.addHandler(log_handler)
    log.setLevel(logging.INFO)
    return log


def configure_mlflow(experiment_name: str) -> None:
    """Set tracking server URL and experiment.

    The MLflow client requires that you pass it the URL of the MLflow
    tracking server and that the client has the appropriate credentials
    to access the object storage for logging models (which in this case
    is via the AWS client library that is used to access object storage
    provided by Minio).
    """
    try:
        mlflow_tracking_uri = os.environ["MLFLOW_TRACKING_URI"]
        if mlflow_tracking_uri == "":
            raise ValueError("MLFLOW_TRACKING_URI cannot be empty")
    except KeyError:
        raise RuntimeError("Cannot find env var MLFLOW_TRACKING_URI.")

    if (
        "MLFLOW_S3_ENDPOINT_URL" in os.environ
        and os.environ["MLFLOW_S3_ENDPOINT_URL"] == "null"
    ):
        del os.environ["MLFLOW_S3_ENDPOINT_URL"]

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)
