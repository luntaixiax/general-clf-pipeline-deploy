import pytest
import pandas as pd
from datetime import date

@pytest.fixture
def client_table() -> pd.DataFrame:
    from src.etl import CustFeatureSnap
    
    CustFeatureSnap.INIT_CUST_SIZE = 10
    CustFeatureSnap.CUST_GROW_SPEED = 2
    CustFeatureSnap.CUST_DIMINISH_SPEED = 1
    
    return CustFeatureSnap.init(date(2024, 1, 1))
    
