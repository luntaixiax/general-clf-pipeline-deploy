import logging
logging.basicConfig(
    format="%(levelname)s | %(asctime)s | %(message)s",
    level = logging.INFO
)
from datetime import date
from luntaiDs.ProviderTools.clickhouse.snap_struct import SnapshotDataManagerCHSQL
from luntaiDs.CommonTools.SnapStructure.dependency import _CurrentStream, _PastStream, _FutureStream
from src.pipeline.utils import SnapTableCH
from src.pipeline.data_extraction import CustomerRaw, AcctRaw, EventRaw, PurchaseRaw

# Features - X
class CustBase(SnapTableCH):
    dm = SnapshotDataManagerCHSQL(schema = 'FEATURE', table = 'CUST_BASE', snap_dt_key = 'SNAP_DT')
    sql_template: str = "src/pipeline/query/transform/feature/cust_base.sql"
    upstreams = [_CurrentStream(CustomerRaw())]

class AcctBase(SnapTableCH):
    dm = SnapshotDataManagerCHSQL(schema = 'FEATURE', table = 'ACCT_BASE', snap_dt_key = 'SNAP_DT')
    sql_template: str = "src/pipeline/query/transform/feature/acct_base.sql"
    upstreams = [_CurrentStream(AcctRaw())]
    
class AcctWindow(SnapTableCH):
    dm = SnapshotDataManagerCHSQL(schema = 'FEATURE', table = 'ACCT_WINDOW', snap_dt_key = 'SNAP_DT')
    sql_template: str = "src/pipeline/query/transform/feature/acct_window.sql"
    upstreams = [_PastStream(AcctBase(), history = 7, freq = 'd')]
    
class EventWindow(SnapTableCH):
    dm = SnapshotDataManagerCHSQL(schema = 'FEATURE', table = 'EVENT_WINDOW', snap_dt_key = 'SNAP_DT')
    sql_template: str = "src/pipeline/query/transform/feature/event_window.sql"
    upstreams = [_PastStream(EventRaw(), history = 7, freq = 'd')]


# Targets - Y
class PurchaseWindow(SnapTableCH):
    dm = SnapshotDataManagerCHSQL(schema = 'TARGET', table = 'PURCHASE_WINDOW', snap_dt_key = 'SNAP_DT')
    sql_template: str = "src/pipeline/query/transform/target/purchase_window.sql"
    upstreams = [_FutureStream(PurchaseRaw(), future = 8, freq = 'd')]