import os
os.environ['ENV'] = 'dev'
import pytest
from datetime import date
from unittest import mock

def test_connection():
    from src.data_connection import S3A, CH_CONF, MONGO, AIRFLOW_API
    
def test_run():
    from src.dag_run import run_pipeline
    
    run_pipeline(task_name='CustFeatureSnap', dag_date=date(2024,1,1))