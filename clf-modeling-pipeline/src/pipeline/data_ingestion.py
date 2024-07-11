import logging
from datetime import date
import pandas as pd
from luntaiDs.ProviderTools.clickhouse.snap_struct import SnapshotDataManagerCHSQL
from luntaiDs.CommonTools.SnapStructure.structure import SnapshotDataManagerObjStorage
from src.dao.data_connection import Connection
from src.pipeline.utils import SnapTableIngestor
from src.utils.settings import ENTITY_CFG

SnapshotDataManagerObjStorage.setup(
    bucket = Connection.DATA_BUCKET,
    root_dir = "/fake/data",
    obja = Connection().S3A
) 

class CustomerRaw(SnapTableIngestor):
    dm = SnapshotDataManagerCHSQL(
        schema = 'RAW', 
        table = 'CUSTOMER', 
        snap_dt_key = ENTITY_CFG.dt_key
    )
    
    @classmethod
    def read(cls, snap_dt: date) -> pd.DataFrame:
        return SnapshotDataManagerObjStorage(
            schema='FEATURES', table='CUST'
        ).read(
            snap_dt,
            columns = [
                "CUST_ID",
                "SNAP_DT",
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
            ]
        ).rename(
            columns = {
                "CUST_ID": ENTITY_CFG.entity_key,
                "SNAP_DT": ENTITY_CFG.dt_key,
            }
        ).reset_index(drop=True) # avoid __index_level_0__ column
        
class AcctRaw(SnapTableIngestor):
    dm = SnapshotDataManagerCHSQL(
        schema = 'RAW', 
        table = 'ACCOUNT', 
        snap_dt_key = ENTITY_CFG.dt_key
    )
    
    @classmethod
    def read(cls, snap_dt: date) -> pd.DataFrame:
        accts =  SnapshotDataManagerObjStorage(
            schema='FEATURES', table='ACCT'
        ).read(
            snap_dt,
            columns = [
                "CUST_ID",
                "ACCT_ID",
                "SNAP_DT",
                "ACCT_TYPE_CD",
                "END_BAL",
                "DR_AMT",
                "CR_AMT",
                "CR_LMT"
            ]
        ).rename(
            columns = {
                "CUST_ID": ENTITY_CFG.entity_key,
                "SNAP_DT": ENTITY_CFG.dt_key,
            }
        ).reset_index(drop=True) # avoid __index_level_0__ column
        return accts
        
class EventRaw(SnapTableIngestor):
    dm = SnapshotDataManagerCHSQL(
        schema = 'RAW', 
        table = 'EVENTS', 
        snap_dt_key = ENTITY_CFG.dt_key
    )
    
    @classmethod
    def read(cls, snap_dt: date) -> pd.DataFrame:
        return SnapshotDataManagerObjStorage(
            schema='EVENTS', table='ENGAGE'
        ).read(
            snap_dt,
            columns = [
                "CUST_ID",
                "SNAP_DT",
                "EVENT_TS",
                "EVENT_CD",
                "EVENT_CHANNEL",
            ]
        ).rename(
            columns = {
                "CUST_ID": ENTITY_CFG.entity_key,
                "SNAP_DT": ENTITY_CFG.dt_key,
            }
        ).reset_index(drop=True) # avoid __index_level_0__ column
        
class PurchaseRaw(SnapTableIngestor):
    dm = SnapshotDataManagerCHSQL(
        schema = 'RAW', 
        table = 'PURCHASES', 
        snap_dt_key = ENTITY_CFG.dt_key
    )
    
    @classmethod
    def read(cls, snap_dt: date) -> pd.DataFrame:
        return SnapshotDataManagerObjStorage(
            schema='EVENTS', table='CONVERSION'
        ).read(
            snap_dt,
            columns = [
                "CUST_ID",
                "SNAP_DT",
                "PUR_TS",
                "PUR_NUM",
                "PUR_AMT",
            ]
        ).rename(
            columns = {
                "CUST_ID": ENTITY_CFG.entity_key,
                "SNAP_DT": ENTITY_CFG.dt_key,
            }
        ).reset_index(drop=True) # avoid __index_level_0__ column
        
