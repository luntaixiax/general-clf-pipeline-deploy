import pytest
import pandas as pd
from datetime import date
from unittest import mock

@pytest.fixture
def client_table() -> pd.DataFrame:
    from src.etl import CustFeatureSnap
    
    CustFeatureSnap.INIT_CUST_SIZE = 10
    CustFeatureSnap.CUST_GROW_SPEED = 2
    CustFeatureSnap.CUST_DIMINISH_SPEED = 1
    
    df = CustFeatureSnap.init(date(2024, 1, 2))
    return df

@pytest.fixture
def acct_table(client_table) -> pd.DataFrame:
    from src.etl import AcctFeatureSnap
    
    with mock.patch('src.etl.CustFeatureSnap.dm') as mock_dm, \
            mock.patch('random.randint') as mock_rnd:
        
        n_rows_client = 10 + 2 - 1
        mock_dm.read_pd.return_value = client_table
        mock_rnd.side_effect = list(range(1, n_rows_client + 1))
        
        df = AcctFeatureSnap.init(date(2024, 1, 1))
    
    return df
    
