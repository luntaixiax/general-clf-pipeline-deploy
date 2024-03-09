import os
os.environ['ENV'] = 'dev'
import pytest
from datetime import date, timedelta
from unittest import mock
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype

@pytest.mark.skipif(os.environ.get('ENV') != 'dev',
                    reason="not correct env")
def test_connection():
    from src.data_connection import S3A, CH_CONF, MONGO, AIRFLOW_API
    
def test_cust_feature(client_table):
    from src.etl import CustFeatureSnap
    
    with mock.patch('src.etl.CustFeatureSnap.dm') as mock_dm, \
            mock.patch('random.randint') as mock_rnd:
            
        mock_dm.read.return_value = client_table
        mock_rnd.return_value = 2
        
        run_dt = date(2024, 1, 1)
        df = CustFeatureSnap.generate(run_dt)
        
        # test if read method has been called
        mock_dm.read.assert_called_once()
        
        assert len(df) == CustFeatureSnap.INIT_CUST_SIZE \
                            + 2 - CustFeatureSnap.CUST_DIMINISH_SPEED
        
        assert is_datetime64_any_dtype(df['SNAP_DT'])
        assert is_datetime64_any_dtype(df['BIRTH_DT'])
        assert is_datetime64_any_dtype(df['SINCE_DT'])