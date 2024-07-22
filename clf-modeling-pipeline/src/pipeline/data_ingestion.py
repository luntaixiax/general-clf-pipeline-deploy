import logging
from datetime import date
import ibis
from ibis import _
from src.pipeline.utils import SnapTableIngestor
from src.utils.settings import ENTITY_CFG


class CustomerRaw(SnapTableIngestor):
    schema = 'RAW'
    table = 'CUSTOMER'
    
    @classmethod
    def read(cls, snap_dt: date) -> ibis.expr.types.Table:
        return (
            cls.get_src_dm(
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
    schema = 'RAW'
    table = 'ACCOUNT'
    
    @classmethod
    def read(cls, snap_dt: date) -> ibis.expr.types.Table:
        accts = (
            cls.get_src_dm(
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
    schema = 'RAW'
    table = 'EVENTS'
    
    @classmethod
    def read(cls, snap_dt: date) -> ibis.expr.types.Table:
        return (
            cls.get_src_dm(
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
    schema = 'RAW'
    table = 'PURCHASES'
    
    @classmethod
    def read(cls, snap_dt: date) -> ibis.expr.types.Table:
        return (
            cls.get_src_dm(
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
        
