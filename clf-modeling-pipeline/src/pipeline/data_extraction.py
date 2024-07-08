import logging
logging.basicConfig(
    format="%(levelname)s | %(asctime)s | %(message)s",
    level = logging.INFO
)
from datetime import date
from luntaiDs.ProviderTools.clickhouse.snap_struct import SnapshotDataManagerCHSQL
from src.dao.data_connection import VAULT_MOUNT_POINT, VAULT_MOUNT_PATH, Connection,\
        get_vault_resp
from src.pipeline.utils import SnapTableCH

def compile_obj_storage_ch_query(file_path: str) -> str:
    response = get_vault_resp(
        mount_point = VAULT_MOUNT_POINT,
        path = VAULT_MOUNT_PATH['s3'],
    )
    endpoint_url = f"http://{response['endpoint']}:{response['port']}"
    if file_path.startswith("/"):
        filepath = endpoint_url + "/" + Connection.DATA_BUCKET + file_path
    else:
        filepath = endpoint_url + "/" + Connection.DATA_BUCKET + "/" + file_path
    if file_path.endswith("parquet"):
        file_fmt = 'Parquet'
    else:
        file_fmt = "CSVWithNames"
        
    return f"s3('{filepath}', '{file_fmt}')"
    
class ObjStorageQuerDateyWrapper:
    """if you want to pass dynamic filepath to add_sql_args (value is a function)"""
    
    def __init__(self, path_tenplate: str):
        """path template that contains {} for date part, e.g., /fake/data/FEATURES/CUST/{}.parquet

        :param str path_tenplate: _description_
        """
        self.path_tenplate = path_tenplate
        
    def __call__(self, snap_dt: date) -> str:
        filepath = self.path_tenplate.format(snap_dt)
        return compile_obj_storage_ch_query(filepath)


class CustomerRaw(SnapTableCH):
    dm = SnapshotDataManagerCHSQL(schema = 'RAW', table = 'CUSTOMER', snap_dt_key = 'SNAP_DT')
    sql_template: str = "src/pipeline/query/ingestion/customer.sql"
    add_sql_args: dict = {
        "obj_storage" : ObjStorageQuerDateyWrapper("/fake/data/FEATURES/CUST/CUST_{}.parquet")
    }
    
class AcctRaw(SnapTableCH):
    dm = SnapshotDataManagerCHSQL(schema = 'RAW', table = 'ACCOUNT', snap_dt_key = 'SNAP_DT')
    sql_template: str = "src/pipeline/query/ingestion/account.sql"
    add_sql_args: dict = {
        "obj_storage" : ObjStorageQuerDateyWrapper("/fake/data/FEATURES/ACCT/ACCT_{}.parquet")
    }
    
class EventRaw(SnapTableCH):
    dm = SnapshotDataManagerCHSQL(schema = 'RAW', table = 'EVENTS', snap_dt_key = 'SNAP_DT')
    sql_template: str = "src/pipeline/query/ingestion/event.sql"
    add_sql_args: dict = {
        "obj_storage" : compile_obj_storage_ch_query('/fake/data/EVENTS/ENGAGE/ENGAGE_*.parquet')
    }
    
    
class PurchaseRaw(SnapTableCH):
    dm = SnapshotDataManagerCHSQL(schema = 'RAW', table = 'PURCHASES', snap_dt_key = 'SNAP_DT')
    sql_template: str = "src/pipeline/query/ingestion/purchase.sql"
    add_sql_args: dict = {
        "obj_storage" : compile_obj_storage_ch_query('/fake/data/EVENTS/CONVERSION/CONVERSION_*.parquet')
    }