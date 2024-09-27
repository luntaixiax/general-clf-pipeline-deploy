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
    from src.data_connection import Connection
    
    c = Connection()
    
def test_cust_feature(client_table):
    
    with mock.patch('src.etl.CustFeatureSnap.dm') as mock_dm, \
            mock.patch('random.randint') as mock_rnd:
                
        from src.etl import CustFeatureSnap
            
        mock_dm.read_pd.return_value = client_table
        mock_rnd.return_value = 2
        
        run_dt = date(2024, 8, 1)
        df = CustFeatureSnap.generate(run_dt)
        
        # test if read method has been called
        mock_dm.read_pd.assert_called_once()
        
        assert len(df) == CustFeatureSnap.INIT_CUST_SIZE \
                            + 2 - CustFeatureSnap.CUST_DIMINISH_SPEED
        
        assert is_datetime64_any_dtype(df['SNAP_DT'])
        assert is_datetime64_any_dtype(df['BIRTH_DT'])
        assert is_datetime64_any_dtype(df['SINCE_DT'])
        
def test_acct_feature(acct_table, client_table):
    
    with mock.patch('src.etl.AcctFeatureSnap.dm') as mock_acct_dm, \
         mock.patch('src.etl.CustFeatureSnap.dm') as mock_client_dm, \
         mock.patch('random.randint') as mock_rnd:
                
        from src.etl import AcctFeatureSnap
            
        mock_acct_dm.read_pd.return_value = acct_table
        mock_client_dm.read_pd.return_value = client_table
        mock_rnd.return_value = 2
        
        run_dt = date(2024, 8, 2)
        df = AcctFeatureSnap.generate(run_dt)
        
        # test if read method has been called
        mock_acct_dm.read_pd.assert_called_once()
        mock_client_dm.read_pd.assert_called_once()
        assert is_datetime64_any_dtype(df['SNAP_DT'])