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
    from src.dao.data_connection import Connection
    
    c = Connection()