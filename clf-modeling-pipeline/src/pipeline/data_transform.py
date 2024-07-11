import ibis
from ibis import _
from datetime import date, timedelta
from luntaiDs.ProviderTools.clickhouse.snap_struct import SnapshotDataManagerCHSQL
from luntaiDs.CommonTools.SnapStructure.dependency import _CurrentStream, _PastStream, _FutureStream
from luntaiDs.CommonTools.SnapStructure.tools import get_past_period_ends
from src.pipeline.utils import static_ibis_arr_avg, static_ibis_arr_max, static_ibis_arr_sum, \
    SnapTableTransfomer
from src.utils.settings import ENTITY_CFG
from src.pipeline.data_extraction import CustomerExtract, AcctExtract, EventExtract, PurchaseExtract
    
# Features - X
class CustBase(SnapTableTransfomer):
    dm = SnapshotDataManagerCHSQL(
        schema = 'FEATURE', 
        table = 'CUST_BASE', 
        snap_dt_key = ENTITY_CFG.dt_key
    )
    upstreams = [_CurrentStream(CustomerExtract())]
    
    @classmethod
    def query(cls, snap_dt: date) -> ibis.expr.types.Table:
        return (
            cls.dm
            .get_table(schema = 'RAW', table = 'CUSTOMER')
            .filter(_[ENTITY_CFG.dt_key] == snap_dt)
            .select(
                ENTITY_CFG.entity_key,
                ENTITY_CFG.dt_key,
                'GENDER',
                EMAIL_DOMAIN = (
                    _['EMAIL']
                    .fill_null('xx@NA')
                    .re_extract('@(\w+)', 1)
                ),
                BLOOD_GRP = _['BLOOD_GRP'],
                SUPER_CITY = (
                    _['OFFICE']
                    .lower()
                    .strip()
                    .isin(['new york', 'los angeles', 'chicago'])
                    .ifelse(True, False)
                ),
                COASTAL_CITY = (
                    _['OFFICE']
                    .lower()
                    .strip()
                    .isin(['new york', 'seattle', 'los angeles'])
                    .ifelse(True, False)
                ),
                MANAGER_FLG = (
                    _['TITLE']
                    .lower()
                    .strip()
                    .isin(['manager', 'senior manager', 'vp'])
                    .ifelse(True, False)
                ),
                TECHNICAL_FLG = (
                    _['ORG']
                    .lower()
                    .strip()
                    .isin(['devops', 'internal tools', 'platform'])
                    .ifelse(True, False)
                ),
                AGE = (
                    - _['BIRTH_DT']
                    .delta(snap_dt, 'year')
                ),
                TENURE = (
                    - _['SINCE_DT']
                    .delta(snap_dt, 'year')
                ),
                TOTAL_COMP = (
                    _['SALARY'].fill_null(0)
                    + _['BONUS'].fill_null(0)
                ),
                BONUS_RATIO = (
                    _['BONUS'].fill_null(0)
                    / (
                        _['SALARY'].fill_null(0)
                        + _['BONUS'].fill_null(0)
                    )
                ).cast('decimal(18,2)')
            )
        )

class AcctBase(SnapTableTransfomer):
    dm = SnapshotDataManagerCHSQL(
        schema = 'FEATURE', 
        table = 'ACCT_BASE', 
        snap_dt_key = ENTITY_CFG.dt_key
    )
    upstreams = [_CurrentStream(AcctExtract())]
    
    @classmethod
    def query(cls, snap_dt: date) -> ibis.expr.types.Table:
        return (
            cls.dm
            .get_table(schema = 'RAW', table = 'ACCOUNT')
            .filter(_[ENTITY_CFG.dt_key] == snap_dt)
            .group_by(ENTITY_CFG.pk)
            .aggregate(
                NUM_ACCTS_DEB = _.nunique(
                    where = (
                        _['ACCT_TYPE_CD']
                        .strip()
                        .isin(['CHQ', 'SAV'])
                    )
                ),
                NUM_ACCTS_CRE = _.nunique(
                    where = (
                        _['ACCT_TYPE_CD']
                        .strip()
                        .notin(['CHQ', 'SAV'])
                    )
                ),
                END_BAL_DEB = _['END_BAL'].sum(
                    where = (
                        _['ACCT_TYPE_CD']
                        .strip()
                        .isin(['CHQ', 'SAV'])
                    )
                ),
                END_BAL_CRE = _['END_BAL'].sum(
                    where = (
                        _['ACCT_TYPE_CD']
                        .strip()
                        .notin(['CHQ', 'SAV'])
                    )
                ),
                DR_AMT_DEB = _['DR_AMT'].sum(
                    where = (
                        _['ACCT_TYPE_CD']
                        .strip()
                        .isin(['CHQ', 'SAV'])
                    )
                ),
                DR_AMT_CRE = _['DR_AMT'].sum(
                    where = (
                        _['ACCT_TYPE_CD']
                        .strip()
                        .notin(['CHQ', 'SAV'])
                    )
                ),
                CR_AMT_DEB = _['CR_AMT'].sum(
                    where = (
                        _['ACCT_TYPE_CD']
                        .strip()
                        .isin(['CHQ', 'SAV'])
                    )
                ),
                CR_AMT_CRE = _['CR_AMT'].sum(
                    where = (
                        _['ACCT_TYPE_CD']
                        .strip()
                        .notin(['CHQ', 'SAV'])
                    )
                ),
                CR_LMT_ = _['CR_LMT'].sum(
                    where = (
                        _['ACCT_TYPE_CD']
                        .strip()
                        .notin(['CHQ', 'SAV'])
                    )
                )
            )
            .rename({'CR_LMT' : 'CR_LMT_'})
        )
    
class AcctWindow(SnapTableTransfomer):
    dm = SnapshotDataManagerCHSQL(
        schema = 'FEATURE', 
        table = 'ACCT_WINDOW', 
        snap_dt_key = ENTITY_CFG.dt_key
    )
    upstreams = [_PastStream(AcctBase(), history = 7, freq = 'D')]
    
    @classmethod
    def query(cls, snap_dt: date) -> ibis.expr.types.Table:
        ACCT_BASE = (
            cls.dm
            .get_table(schema = 'FEATURE', table = 'ACCT_BASE')
        )
        CATESIAN = (
            ACCT_BASE
            .filter(_[ENTITY_CFG.dt_key] == snap_dt)
            .mutate(
                DT = ibis.array(
                    get_past_period_ends(snap_dt, history = 7, freq = 'D')[::-1]
                ).unnest()
            )
            .select(ENTITY_CFG.entity_key, 'DT')
        )
        WINDOW = (
            CATESIAN
            .left_join(
                ACCT_BASE,
                [
                    CATESIAN[ENTITY_CFG.entity_key] == ACCT_BASE[ENTITY_CFG.entity_key],
                    CATESIAN['DT'] == ACCT_BASE[ENTITY_CFG.dt_key]
                ]
            )
            .select(
                ENTITY_CFG.entity_key,
                _['DT'].name(ENTITY_CFG.dt_key),
                _['NUM_ACCTS_DEB'].fill_null(0).name('NUM_ACCTS_DEB'),
                _['NUM_ACCTS_CRE'].fill_null(0).name('NUM_ACCTS_CRE'),
                (
                    _['NUM_ACCTS_DEB'].fill_null(0)
                    + _['NUM_ACCTS_CRE'].fill_null(0)
                ).name('NUM_ACCTS'),
                _['END_BAL_DEB'].fill_null(0).name('END_BAL_DEB'),
                _['END_BAL_CRE'].fill_null(0).name('END_BAL_CRE'),
                _['DR_AMT_DEB'].fill_null(0).name('DR_AMT_DEB'),
                _['DR_AMT_CRE'].fill_null(0).name('DR_AMT_CRE'),
                _['CR_AMT_DEB'].fill_null(0).name('CR_AMT_DEB'),
                _['CR_AMT_CRE'].fill_null(0).name('CR_AMT_CRE'),
                (
                    _['DR_AMT_DEB'].fill_null(0)
                    - _['CR_AMT_DEB'].fill_null(0)
                ).name('TOTAL_DEB_FLOW'),
                (
                    _['CR_AMT_CRE'].fill_null(0)
                    - _['DR_AMT_CRE'].fill_null(0)
                ).name('TOTAL_CRE_FLOW'),
                (
                    _['DR_AMT_DEB'].fill_null(0)
                    - _['CR_AMT_DEB'].fill_null(0)
                    + _['CR_AMT_CRE'].fill_null(0)
                    - _['DR_AMT_CRE'].fill_null(0)
                ).name('TOTAL_FLOW'),
                _['CR_LMT'].fill_null(0).name('CR_LMT'),
            )
            .order_by([ENTITY_CFG.entity_key, ibis.desc(ENTITY_CFG.dt_key)])
            .group_by(ENTITY_CFG.entity_key)
            .aggregate(
                # past 7d features
                _['NUM_ACCTS'].collect()[:7].name('NUM_ACCTS'),
                _['TOTAL_FLOW'].collect()[:7].name('TOTAL_FLOW'),
                _['TOTAL_DEB_FLOW'].collect()[:7].name('TOTAL_DEB_FLOW'),
                _['TOTAL_CRE_FLOW'].collect()[:7].name('TOTAL_CRE_FLOW'),
                _['END_BAL_CRE'].cast('float').collect()[:7].name('END_BAL_CRE_7D'),
                _['CR_LMT'].collect()[:7].name('CR_LMT'),
                # past 3d features
                _['DR_AMT_DEB'].collect()[:3].name('DR_AMT_DEB'),
                _['CR_AMT_DEB'].collect()[:3].name('CR_AMT_DEB'),
                _['END_BAL_DEB'].cast('float').collect()[:3].name('END_BAL_DEB'),
                _['DR_AMT_CRE'].collect()[:3].name('DR_AMT_CRE'),
                _['CR_AMT_CRE'].collect()[:3].name('CR_AMT_CRE'),
                _['END_BAL_CRE'].cast('float').collect()[:3].name('END_BAL_CRE_3D'),
            )
        )
        return (
            WINDOW
            .group_by(ENTITY_CFG.entity_key)
            .mutate(
                # past 7d features
                static_ibis_arr_max(_['NUM_ACCTS'], 7).name('MAX_NUM_ACCT_7D'),
                static_ibis_arr_sum(_['TOTAL_FLOW'], 7).name('TOTAL_FLOW_7D'),
                static_ibis_arr_sum(_['TOTAL_DEB_FLOW'], 7).name('TOTAL_DEB_FLOW_7D'),
                static_ibis_arr_sum(_['TOTAL_CRE_FLOW'], 7).name('TOTAL_CRE_FLOW_7D'),
                static_ibis_arr_avg(_['END_BAL_CRE_7D'], 7).name('AVG_END_BAL_CRE_7D'),
                static_ibis_arr_max(_['END_BAL_CRE_7D'], 7).name('MAX_END_BAL_CRE_7D'),
                static_ibis_arr_max(_['CR_LMT'], 7).name('MAX_CR_LMT_7D'),
                # past 3d features
                static_ibis_arr_sum(_['DR_AMT_DEB'], 3).name('TOTAL_DR_AMT_DEB_3D'),
                static_ibis_arr_sum(_['CR_AMT_DEB'], 3).name('TOTAL_CR_AMT_DEB_3D'),
                static_ibis_arr_avg(_['END_BAL_DEB'], 3).name('AVG_END_BAL_DEB_3D'),
                static_ibis_arr_sum(_['DR_AMT_CRE'], 3).name('TOTAL_DR_AMT_CRE_3D'),
                static_ibis_arr_sum(_['CR_AMT_CRE'], 3).name('TOTAL_CR_AMT_CRE_3D'),
                static_ibis_arr_avg(_['END_BAL_CRE_3D'], 3).name('AVG_END_BAL_CRE_3D'),
            )
            .select(
                ENTITY_CFG.entity_key,
                ibis.literal(snap_dt).name(ENTITY_CFG.dt_key),
                'MAX_NUM_ACCT_7D',
                'TOTAL_FLOW_7D',
                (
                    _['TOTAL_DEB_FLOW_7D'].cast('float') / _['TOTAL_CRE_FLOW_7D'].nullif(0)
                ).fill_null(0).name('DEB_CRE_FLOW_7D'),
                (
                    _['AVG_END_BAL_CRE_7D'].cast('float') / _['MAX_CR_LMT_7D'].nullif(0)
                ).fill_null(0).name('AVG_CR_UTIL_7D'),
                (
                    _['MAX_END_BAL_CRE_7D'].cast('float') / _['MAX_CR_LMT_7D'].nullif(0)
                ).fill_null(0).name('MAX_CR_UTIL_7D'),
                (
                    _['TOTAL_DR_AMT_DEB_3D'].cast('float') / _['AVG_END_BAL_DEB_3D'].nullif(0)
                ).fill_null(0).name('AVG_DR_DEB_3D'),
                (
                    _['TOTAL_CR_AMT_DEB_3D'].cast('float') / _['AVG_END_BAL_DEB_3D'].nullif(0)
                ).fill_null(0).name('AVG_CR_DEB_3D'),
                (
                    _['TOTAL_DR_AMT_CRE_3D'].cast('float') / _['AVG_END_BAL_CRE_3D'].nullif(0)
                ).fill_null(0).name('AVG_DR_CRE_3D'),
                (
                    _['TOTAL_CR_AMT_CRE_3D'].cast('float') / _['AVG_END_BAL_CRE_3D'].nullif(0)
                ).fill_null(0).name('AVG_CR_CRE_3D'),
            )
        )
    
class EventWindow(SnapTableTransfomer):
    dm = SnapshotDataManagerCHSQL(
        schema = 'FEATURE', 
        table = 'EVENT_WINDOW', 
        snap_dt_key = ENTITY_CFG.dt_key
    )
    upstreams = [_PastStream(EventExtract(), history = 7, freq = 'D')]
    
    @classmethod
    def query(cls, snap_dt: date) -> ibis.expr.types.Table:
        return (
            cls.dm
            .get_table(schema = 'RAW', table = 'EVENTS')
            .filter(
                _['EVENT_TS'].between(
                    snap_dt - timedelta(days = 7),
                    snap_dt
                )
            )
            .group_by(ENTITY_CFG.entity_key)
            .aggregate(
                _['EVENT_TS'].max().name('MAX_EVENT_TS'),
                _['EVENT_TS'].count().name('NUM_EVENT_7D'),
                _.count(
                    where = _['EVENT_CHANNEL'].strip() == 'E'
                ).name('NUM_EVENT_EMAIL'),
                _.count(
                    where = _['EVENT_CD'].strip() == 'C'
                ).name('NUM_EVENT_CLICK'),
                _.count(
                    where = _['EVENT_CD'].strip() == 'D'
                ).name('NUM_EVENT_DECLINE')
            )
            .select(
                ENTITY_CFG.entity_key,
                ibis.literal(snap_dt).name(ENTITY_CFG.dt_key),
                (
                    - _['MAX_EVENT_TS']
                    .cast('date')
                    .delta(snap_dt, 'day')
                ).name('DAYS_SINCE_LAST_EVENT'),
                'NUM_EVENT_7D',
                (_['NUM_EVENT_EMAIL'] / _['NUM_EVENT_7D']).name('RATIO_EVENT_EMAIL'),
                (_['NUM_EVENT_CLICK'] / _['NUM_EVENT_7D']).name('RATIO_EVENT_CLICK'),
                (_['NUM_EVENT_DECLINE'] / _['NUM_EVENT_7D']).name('RATIO_EVENT_DECLINE')
            )
        )

class Features(SnapTableTransfomer):
    dm = SnapshotDataManagerCHSQL(
        schema = 'FEATURE', 
        table = 'FEATURES', 
        snap_dt_key = ENTITY_CFG.dt_key
    )
    upstreams = [_CurrentStream(CustBase()), _CurrentStream(AcctWindow()),
                 _CurrentStream(EventWindow())]
    
    @classmethod
    def query(cls, snap_dt: date) -> ibis.expr.types.Table:
        # TODO: rename because clickhouse will not remove table prefix
        CUST_BASE = (
            cls.dm
            .get_table(schema = 'FEATURE', table = 'CUST_BASE')
            .filter(_[ENTITY_CFG.dt_key] == snap_dt)
        )
        CUST_BASE = CUST_BASE.rename({
            c + "_" : c
            for c in CUST_BASE.columns
        })

        ACCT_WINDOW = (
            cls.dm
            .get_table(schema = 'FEATURE', table = 'ACCT_WINDOW')
            .filter(_[ENTITY_CFG.dt_key] == snap_dt)
        )
        ACCT_WINDOW = ACCT_WINDOW.rename({
            c + "_" : c
            for c in ACCT_WINDOW.columns
        })
        EVENT_WINDOW = (
            cls.dm
            .get_table(schema = 'FEATURE', table = 'EVENT_WINDOW')
            .filter(_[ENTITY_CFG.dt_key] == snap_dt)
        )
        EVENT_WINDOW = EVENT_WINDOW.rename({
            c + "_" : c
            for c in EVENT_WINDOW.columns
        })
        
        return (
            CUST_BASE
            .left_join(
                ACCT_WINDOW, 
                [ENTITY_CFG.entity_key + "_", ENTITY_CFG.dt_key + "_"],
                rname = '{name}_acct'
            )
            .left_join(
                EVENT_WINDOW, 
                [ENTITY_CFG.entity_key + "_", ENTITY_CFG.dt_key + "_"],
                rname = '{name}_event'
            )
            .select(
                _[ENTITY_CFG.entity_key + "_"].name(ENTITY_CFG.entity_key),
                _[ENTITY_CFG.dt_key + "_"].name(ENTITY_CFG.dt_key),
                # customer feature
                _['GENDER_'].name('GENDER'),
                _['EMAIL_DOMAIN_'].name('EMAIL_DOMAIN'),
                _['BLOOD_GRP_'].name('BLOOD_GRP'),
                _['SUPER_CITY_'].name('SUPER_CITY'),
                _['COASTAL_CITY_'].name('COASTAL_CITY'),
                _['MANAGER_FLG_'].name('MANAGER_FLG'),
                _['TECHNICAL_FLG_'].name('TECHNICAL_FLG'),
                _['AGE_'].name('AGE'),
                _['TENURE_'].name('TENURE'),
                _['TOTAL_COMP_'].name('TOTAL_COMP'),
                # account feature
                _['MAX_NUM_ACCT_7D_'].name('MAX_NUM_ACCT_7D'),
                _['TOTAL_FLOW_7D_'].name('TOTAL_FLOW_7D'),
                _['DEB_CRE_FLOW_7D_'].name('DEB_CRE_FLOW_7D'),
                _['AVG_CR_UTIL_7D_'].name('AVG_CR_UTIL_7D'),
                _['MAX_CR_UTIL_7D_'].name('MAX_CR_UTIL_7D'),
                _['AVG_DR_DEB_3D_'].name('AVG_DR_DEB_3D'),
                _['AVG_CR_DEB_3D_'].name('AVG_CR_DEB_3D'),
                _['AVG_DR_CRE_3D_'].name('AVG_DR_CRE_3D'),
                _['AVG_CR_CRE_3D_'].name('AVG_CR_CRE_3D'),
                # event features
                _['DAYS_SINCE_LAST_EVENT_'].fill_null(0).name('DAYS_SINCE_LAST_EVENT'),
                _['NUM_EVENT_7D_'].fill_null(0).name('NUM_EVENT_7D'),
                _['RATIO_EVENT_EMAIL_'].name('RATIO_EVENT_EMAIL'), # allow null
                _['RATIO_EVENT_CLICK_'].name('RATIO_EVENT_CLICK'), # allow null
                _['RATIO_EVENT_DECLINE_'].name('RATIO_EVENT_DECLINE'), # allow null
            )
        )

# Targets - Y
class PurchaseWindow(SnapTableTransfomer):
    dm = SnapshotDataManagerCHSQL(
        schema = 'TARGET', 
        table = 'PURCHASE_WINDOW', 
        snap_dt_key = ENTITY_CFG.dt_key
    )
    upstreams = [_FutureStream(PurchaseExtract(), offset = 1, future = 7, freq = 'D')]
    
    @classmethod
    def query(cls, snap_dt: date) -> ibis.expr.types.Table:
        return (
            cls.dm
            .get_table(schema = 'RAW', table = 'PURCHASES')
            .filter(
                _['PUR_TS'].between(
                    snap_dt + timedelta(days = 1),
                    snap_dt + timedelta(days = 8)
                )
            )
            .group_by(ENTITY_CFG.entity_key)
            .aggregate(
                _['PUR_TS'].count().name('NUM_PUR_F7D'),
                _['PUR_NUM'].sum().name('TOTAL_PUR_NUM_F7D'),
                _['PUR_AMT'].sum().name('PUR_AMT'),
            )
            .select(
                ENTITY_CFG.entity_key,
                ibis.literal(snap_dt).name(ENTITY_CFG.dt_key),
                'NUM_PUR_F7D',
                'TOTAL_PUR_NUM_F7D',
                'PUR_AMT'
            )
        )