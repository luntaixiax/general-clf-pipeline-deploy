import ibis
from ibis import _
from datetime import date
from luntaiDs.ProviderTools.clickhouse.snap_struct import SnapshotDataManagerCHSQL
from luntaiDs.CommonTools.SnapStructure.dependency import _CurrentStream, _PastStream, _FutureStream
from src.dao.data_connection import VAULT_MOUNT_POINT, VAULT_MOUNT_PATH, Connection,\
        get_vault_resp
from src.pipeline.utils import SnapTableTransfomer
from src.utils.settings import ENTITY_CFG
from src.pipeline.data_ingestion import CustomerRaw, AcctRaw, EventRaw, PurchaseRaw

class CustomerExtract(SnapTableTransfomer):
    dm = SnapshotDataManagerCHSQL(
        schema = 'EXTRACT', 
        table = 'CUSTOMER',
        snap_dt_key = ENTITY_CFG.dt_key
    )
    upstreams = [_CurrentStream(CustomerRaw())]
    
    @classmethod
    def query(cls, snap_dt: date) -> ibis.expr.types.Table:
        return (
            cls.dm
            .get_table(schema = 'RAW', table = 'CUSTOMER')
            .filter(_[ENTITY_CFG.dt_key] == snap_dt)
        )
    
    
class AcctExtract(SnapTableTransfomer):
    dm = SnapshotDataManagerCHSQL(
        schema = 'EXTRACT', 
        table = 'ACCOUNT', 
        snap_dt_key = ENTITY_CFG.dt_key
    )
    upstreams = [_CurrentStream(AcctRaw())]
    
    @classmethod
    def query(cls, snap_dt: date) -> ibis.expr.types.Table:
        return (
            cls.dm
            .get_table(schema = 'RAW', table = 'ACCOUNT')
            .filter(_[ENTITY_CFG.dt_key] == snap_dt)
            .select(
                'ACCT_ID',
                ENTITY_CFG.entity_key,
                ENTITY_CFG.dt_key,
                'ACCT_TYPE_CD',
                'DR_AMT',
                'CR_AMT',
                'END_BAL',
                'CR_LMT'
            )
        )
    
class EventExtract(SnapTableTransfomer):
    dm = SnapshotDataManagerCHSQL(
        schema = 'EXTRACT', 
        table = 'EVENTS', 
        snap_dt_key = ENTITY_CFG.dt_key
    )
    # need events generated on fake between [t-7, t-1]
    upstreams = [_PastStream(EventRaw(), history=7, offset=-1, freq='D')]
    
    @classmethod
    def query(cls, snap_dt: date) -> ibis.expr.types.Table:
        return (
            cls.dm
            .get_table(schema = 'RAW', table = 'EVENTS')
            # extract only events happened on snap_dt
            .filter(_['EVENT_TS'].cast('date') == snap_dt)
            .select(
                ENTITY_CFG.entity_key,
                ibis.literal(snap_dt).name(ENTITY_CFG.dt_key),
                'EVENT_TS',
                'EVENT_CHANNEL',
                'EVENT_CD'
            )
        )
    
    
class PurchaseExtract(SnapTableTransfomer):
    dm = SnapshotDataManagerCHSQL(
        schema = 'EXTRACT', 
        table = 'PURCHASES', 
        snap_dt_key = ENTITY_CFG.dt_key
    )
    # need events generated on fake between [t-7, t-1]
    upstreams = [_PastStream(PurchaseRaw(), history=7, offset=-1, freq='D')]
    
    @classmethod
    def query(cls, snap_dt: date) -> ibis.expr.types.Table:
        return (
            cls.dm
            .get_table(schema = 'RAW', table = 'PURCHASES')
            # extract only events happened on snap_dt
            .filter(_['PUR_TS'].cast('date') == snap_dt)
            .select(
                ENTITY_CFG.entity_key,
                ibis.literal(snap_dt).name(ENTITY_CFG.dt_key),
                'PUR_TS',
                'PUR_AMT',
                'PUR_NUM'
            )
        )
    