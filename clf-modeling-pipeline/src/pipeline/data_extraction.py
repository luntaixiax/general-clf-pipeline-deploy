import ibis
from ibis import _
from datetime import date
from luntaiDs.CommonTools.SnapStructure.dependency import _CurrentStream, _PastStream
from src.pipeline.utils import SnapTableTransfomer
from src.utils.settings import ENTITY_CFG
from src.pipeline.data_ingestion import CustomerRaw, AcctRaw, EventRaw, PurchaseRaw

class CustomerExtract(SnapTableTransfomer):
    schema = 'EXTRACT'
    table = 'CUSTOMER'
    upstreams = [_CurrentStream(CustomerRaw())]
    
    @classmethod
    def query(cls, snap_dt: date) -> ibis.expr.types.Table:
        return (
            cls.dm
            .get_table(schema = 'RAW', table = 'CUSTOMER')
            .filter(_[ENTITY_CFG.dt_key] == snap_dt)
        )
    
    
class AcctExtract(SnapTableTransfomer):
    schema = 'EXTRACT'
    table = 'ACCOUNT'
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
    schema = 'EXTRACT'
    table = 'EVENTS'
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
    schema = 'EXTRACT'
    table = 'PURCHASES'
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
    