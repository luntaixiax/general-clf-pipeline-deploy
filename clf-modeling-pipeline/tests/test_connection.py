import os
os.environ['ENV'] = 'dev'

def test_connect():
    from src.data_connection import S3A, CH_CONF, MONGO, AIRFLOW_API
    
def test_import():
    from src.pipeline.data_extraction import compile_obj_storage_ch_query