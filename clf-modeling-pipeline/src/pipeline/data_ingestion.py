import logging
from datetime import date
import ibis
from ibis import _
from luntaiDs.ProviderTools.clickhouse.snap_struct import SnapshotDataManagerCHSQL
from luntaiDs.CommonTools.SnapStructure.structure import SnapshotDataManagerFileSystem
from src.dao.data_connection import Connection
from src.pipeline.utils import SnapTableIngestor
from src.utils.settings import ENTITY_CFG

SnapshotDataManagerFileSystem.setup(
    fs = Connection().FS_STORAGE,
    root_dir = f"{Connection.DATA_BUCKET}/fake/data",
) 

class CustomerRaw(SnapTableIngestor):
    dm = SnapshotDataManagerCHSQL(
        schema = 'RAW', 
        table = 'CUSTOMER', 
        snap_dt_key = ENTITY_CFG.dt_key
    )
    
    @classmethod
    def read(cls, snap_dt: date) -> ibis.expr.types.Table:
        return (
            SnapshotDataManagerFileSystem(
                schema='FEATURES', table='CUST'
            )
            .read(snap_dt)
            .select(
                _["CUST_ID"].name(ENTITY_CFG.entity_key),
                _["SNAP_DT"].name(ENTITY_CFG.dt_key),
                "NAME",
                "GENDER",
                "BIRTH_DT",
                "PHONE",
                "EMAIL",
                "BLOOD_GRP",
                "JOB",
                "OFFICE",
                "ADDRESS",
                "ORG",
                "TITLE",
                "SINCE_DT",
                "SALARY",
                "BONUS"
            )
        )
        
class AcctRaw(SnapTableIngestor):
    dm = SnapshotDataManagerCHSQL(
        schema = 'RAW', 
        table = 'ACCOUNT', 
        snap_dt_key = ENTITY_CFG.dt_key
    )
    
    @classmethod
    def read(cls, snap_dt: date) -> ibis.expr.types.Table:
        accts = (
            SnapshotDataManagerFileSystem(
                schema='FEATURES', table='ACCT'
            )
            .read(snap_dt)
            .select(
                _["CUST_ID"].name(ENTITY_CFG.entity_key),
                "ACCT_ID",
                _["SNAP_DT"].name(ENTITY_CFG.dt_key),
                "ACCT_TYPE_CD",
                "END_BAL",
                "DR_AMT",
                "CR_AMT",
                "CR_LMT"
            )
        )
        return accts
        
class EventRaw(SnapTableIngestor):
    dm = SnapshotDataManagerCHSQL(
        schema = 'RAW', 
        table = 'EVENTS', 
        snap_dt_key = ENTITY_CFG.dt_key
    )
    
    @classmethod
    def read(cls, snap_dt: date) -> ibis.expr.types.Table:
        return (
            SnapshotDataManagerFileSystem(
                schema='EVENTS', table='ENGAGE'
            )
            .read(snap_dt)
            .select(
                _["CUST_ID"].name(ENTITY_CFG.entity_key),
                _["SNAP_DT"].name(ENTITY_CFG.dt_key),
                "EVENT_TS",
                "EVENT_CD",
                "EVENT_CHANNEL",
            )
        )
        
        
class PurchaseRaw(SnapTableIngestor):
    dm = SnapshotDataManagerCHSQL(
        schema = 'RAW', 
        table = 'PURCHASES', 
        snap_dt_key = ENTITY_CFG.dt_key
    )
    
    @classmethod
    def read(cls, snap_dt: date) -> ibis.expr.types.Table:
        return (
            SnapshotDataManagerFileSystem(
                schema='EVENTS', table='CONVERSION'
            )
            .read(snap_dt)
            .select(
                _["CUST_ID"].name(ENTITY_CFG.entity_key),
                _["SNAP_DT"].name(ENTITY_CFG.dt_key),
                "PUR_TS",
                "PUR_NUM",
                "PUR_AMT",
            )
        )
        
