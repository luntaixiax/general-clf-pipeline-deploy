import logging
logging.basicConfig(level=logging.INFO)
from datetime import date, timedelta
import random
import pandas as pd
import numpy as np
from luntaiDs.CommonTools.SnapStructure.dependency import SnapTableStreamGenerator, \
    _CurrentStream, _FutureStream, _PastStream, ExecPlan
from luntaiDs.CommonTools.SnapStructure.structure import SnapshotDataManagerFileSystem
from luntaiDs.CommonTools.utils import dt2str, str2dt

from src.data_connection import Connection
from src.model import CustFeature, AcctFeature, lognormal2normal
from src.registry import load_fake_model_by_timetable


SnapshotDataManagerFileSystem.setup(
    fs = Connection().FS_STORAGE,
    root_dir = f"{Connection.DATA_BUCKET}/fake/data",
) 

class CustFeatureSnap(SnapTableStreamGenerator):
    dm = SnapshotDataManagerFileSystem(schema='FEATURES', table='CUST')
    # generator param
    INIT_DT = date(2024, 1, 1)
    INIT_CUST_SIZE = 10000
    CUST_GROW_SPEED = 50 # per day
    CUST_DIMINISH_SPEED = 5 # per day

    @classmethod
    def execute(cls, snap_dt: date):
        if snap_dt < cls.INIT_DT:
            raise ValueError(f"cannot run before init date @ {cls.INIT_DT}")

        logging.info(f"Executing Cust Data @ {snap_dt}")
        existing_dts = cls.dm.get_existing_snap_dts()
        if snap_dt == cls.INIT_DT:
            logging.info(f"Initializing Cust Data @ {snap_dt}...")
            cust_df = cls.init(snap_dt)
            cls.dm.save(cust_df, snap_dt, overwrite=True)
        else:
            last_snap_dt = snap_dt - timedelta(days=1)
            if last_snap_dt not in existing_dts:
                cls.execute(last_snap_dt)

            logging.info(f"Generating Cust Data @ {snap_dt}...")
            cust_df = cls.generate(snap_dt)
            cls.dm.save(cust_df, snap_dt, overwrite=True)

    @classmethod
    def init(cls, snap_dt: date) -> pd.DataFrame:
        custs = []
        for _ in range(cls.INIT_CUST_SIZE):
            cust = CustFeature.new(snap_dt)
            custs.append(cust)

        custs = pd.DataFrame.from_records(custs)
        custs['SNAP_DT'] = pd.to_datetime(custs['SNAP_DT'], utc = True)
        custs['BIRTH_DT'] = pd.to_datetime(custs['BIRTH_DT'], utc = True)
        custs['SINCE_DT'] = pd.to_datetime(custs['SINCE_DT'], utc = True)

        return custs

    @classmethod
    def generate(cls, snap_dt: date) -> pd.DataFrame:
        last_custs = cls.dm.read_pd(snap_dt=snap_dt - timedelta(days=1))
        last_custs = last_custs.to_dict(orient='records')

        new_custs = []
        # augment last records
        for last_cust in last_custs:
            new_cust = CustFeature.variate(snap_dt, last_cust)
            new_custs.append(new_cust)

        # generate new cust
        num_new_custs = random.randint(
            int(cls.CUST_GROW_SPEED * 0.4),
            int(cls.CUST_GROW_SPEED * 1.5)
        )

        for _ in range(num_new_custs):
            new_cust = CustFeature.new(snap_dt)
            new_custs.append(new_cust)

        new_custs = pd.DataFrame.from_records(new_custs)
        new_custs['SNAP_DT'] = pd.to_datetime(new_custs['SNAP_DT'], utc = True)
        new_custs['BIRTH_DT'] = pd.to_datetime(new_custs['BIRTH_DT'], utc = True)
        new_custs['SINCE_DT'] = pd.to_datetime(new_custs['SINCE_DT'], utc = True)
        # randomly drop records, for diminishing
        custs_drop = new_custs['CUST_ID'].sample(cls.CUST_DIMINISH_SPEED)
        new_custs = new_custs[~new_custs['CUST_ID'].isin(custs_drop)].reset_index(drop=True) 

        return new_custs

    @classmethod
    def if_success(cls, snap_dt: date) -> bool:
        return cls.dm.count(snap_dt) > 0

class AcctFeatureSnap(SnapTableStreamGenerator):
    dm = SnapshotDataManagerFileSystem(schema='FEATURES', table='ACCT')
    upstreams = [_CurrentStream(CustFeatureSnap())]
    # generator param
    INIT_DT = date(2024, 1, 1)
    
    @classmethod
    def execute(cls, snap_dt: date):
        if snap_dt < cls.INIT_DT:
            raise ValueError(f"cannot run before init date @ {cls.INIT_DT}")

        logging.info(f"Executing Acct Data @ {snap_dt}")
        existing_dts = cls.dm.get_existing_snap_dts()
        if snap_dt == cls.INIT_DT:
            logging.info(f"Initializing Acct Data @ {snap_dt}...")
            acct_df = cls.init(snap_dt)
            cls.dm.save(acct_df, snap_dt, overwrite=True)
        else:
            last_snap_dt = snap_dt - timedelta(days=1)
            if last_snap_dt not in existing_dts:
                cls.execute(last_snap_dt)

            logging.info(f"Generating Acct Data @ {snap_dt}...")
            acct_df = cls.generate(snap_dt)
            cls.dm.save(acct_df, snap_dt, overwrite=True)

    @classmethod
    def init(cls, snap_dt: date) -> pd.DataFrame:
        sdp_cust = CustFeatureSnap.dm
        custs = sdp_cust.read_pd(snap_dt)

        accts = []
        for idx, cust in custs.iterrows():
            num_acct = random.randint(1, 5)
            for _ in range(num_acct):
                acct = AcctFeature.new(snap_dt)
                acct['CUST_ID'] = cust['CUST_ID']  # link customer with accounts
                accts.append(acct)

        accts = pd.DataFrame.from_records(accts)
        accts['SNAP_DT'] = pd.to_datetime(accts['SNAP_DT'], utc = True)

        return accts

    @classmethod
    def generate(cls, snap_dt: date) -> pd.DataFrame:
        last_accts = cls.dm.read_pd(snap_dt=snap_dt - timedelta(days=1))

        new_accts = []
        # augment last records
        for last_acct in last_accts.to_dict(orient='records'):
            new_acct = AcctFeature.variate(snap_dt, last_acct)
            new_accts.append(new_acct)

        # generate new accts for new custs
        sdp_cust = CustFeatureSnap.dm
        custs = sdp_cust.read_pd(snap_dt)
        new_custs = custs[~custs['CUST_ID'].isin(last_accts['CUST_ID'])]
        for idx, new_cust in new_custs.iterrows():
            num_acct = random.randint(1, 5)
            for _ in range(num_acct):
                new_acct = AcctFeature.new(snap_dt)
                new_acct['CUST_ID'] = new_cust['CUST_ID']  # link customer with accounts
                new_accts.append(new_acct)

        new_accts = pd.DataFrame.from_records(new_accts)
        new_accts['SNAP_DT'] = pd.to_datetime(new_accts['SNAP_DT'], utc = True)
        # only keep same customers as in cust table
        new_accts = new_accts[new_accts['CUST_ID'].isin(custs['CUST_ID'])]

        return new_accts

    @classmethod
    def if_success(cls, snap_dt: date) -> bool:
        return cls.dm.count(snap_dt) > 0


class EngagementFeatureSnap(SnapTableStreamGenerator):
    dm = SnapshotDataManagerFileSystem(schema='FEATURES', table='ENGAGE_FE')
    upstreams = [
        _CurrentStream(CustFeatureSnap()),
        _PastStream(AcctFeatureSnap(), history=7, freq='d')
    ]

    @classmethod
    def execute(cls, snap_dt: date):
        sdp_cust = CustFeatureSnap.dm
        sdp_acct = AcctFeatureSnap.dm
        # get feature
        cust_past = sdp_cust.read_pd(snap_dt)  # customer feature
        past_7ds = [snap_dt - timedelta(days=d) for d in range(7)]
        accts_past_7d = sdp_acct.reads_pd(past_7ds)  # account feature

        # feature engineering
        cust_past['AGE'] = (cust_past['SNAP_DT'] - cust_past['BIRTH_DT']).dt.days / 365
        cust_past['TENURE'] = (cust_past['SNAP_DT'] - cust_past['SINCE_DT']).dt.days / 365
        cust_past['TOTAL_COMP'] = cust_past['SALARY'].fillna(0) + cust_past['BONUS'].fillna(0)
        cust_past['BONUS_RATIO'] = cust_past['BONUS'].fillna(0) / cust_past['TOTAL_COMP']
        cust_past = cust_past.drop(columns=['JOB', 'EMAIL', 'PHONE', 'NAME', 'ADDRESS',
                                            'BIRTH_DT', 'SINCE_DT', 'SALARY', 'BONUS'])

        # transform acct feature
        accts_past_7d.loc[accts_past_7d['ACCT_TYPE_CD'].isin(['CHQ', 'SAV']), 'DIRECTION'] = 'DR'
        accts_past_7d.loc[~accts_past_7d['ACCT_TYPE_CD'].isin(['CHQ', 'SAV']), 'DIRECTION'] = 'CR'
        # accts_past_7d['END_BAL_DIR'] = accts_past_7d['DIRECTION'] * accts_past_7d['END_BAL']
        # accts_past_7d['DR_AMT_DIR'] = accts_past_7d['DIRECTION'] * accts_past_7d['DR_AMT']
        # accts_past_7d['CR_AMT_DIR'] = accts_past_7d['DIRECTION'] * accts_past_7d['CR_AMT']
        # accts_past_7d.loc[~accts_past_7d['ACCT_TYPE_CD'].isin(['CHQ', 'SAV']), 'CR_LMT_DIR'] = 1
        accts_past_7d = (
            accts_past_7d
            .groupby(by=['CUST_ID', 'DIRECTION', 'SNAP_DT'])  # agg account dimension
            .agg({
                'ACCT_ID': 'count',
                'END_BAL': 'sum',
                'DR_AMT': 'sum',
                'CR_AMT': 'sum',
                'CR_LMT': 'sum'
            })
            .groupby(by=['CUST_ID', 'DIRECTION'])
            .agg({
                'ACCT_ID': 'max',  # max no of accounts over past 7d
                'END_BAL': ['max', 'mean'],  # max/mean balance over past 7d
                'DR_AMT': ['max', 'sum'],  # max/total debit amount over past 7d
                'CR_AMT': ['max', 'sum'],  # max/total credit amount over past 7d
                'CR_LMT': 'max'  # max credit limit over past 7d
            })
            # .reset_index()
        )

        accts_past_7d['NUM_ACCTS'] = accts_past_7d['ACCT_ID']['max']
        accts_past_7d['CR_UTIL'] = np.where(
            accts_past_7d['CR_LMT']['max'] != 0,
            accts_past_7d['END_BAL']['mean'] / accts_past_7d['CR_LMT']['max'],
            np.nan
        )
        accts_past_7d['MAX_CR_UTIL'] = np.where(
            accts_past_7d['CR_LMT']['max'] != 0,
            accts_past_7d['END_BAL']['max'] / accts_past_7d['CR_LMT']['max'],
            np.nan
        )

        accts_past_7d['TOTAL_DR_RATIO'] = np.where(
            accts_past_7d['END_BAL']['mean'] != 0,
            accts_past_7d['DR_AMT']['sum'] / accts_past_7d['END_BAL']['mean'],
            np.nan
        )
        accts_past_7d['MAX_DR_RATIO'] = np.where(
            accts_past_7d['END_BAL']['max'] != 0,
            accts_past_7d['DR_AMT']['max'] / accts_past_7d['END_BAL']['max'],
            np.nan
        )
        accts_past_7d['TOTAL_CR_RATIO'] = np.where(
            accts_past_7d['END_BAL']['mean'] != 0,
            accts_past_7d['CR_AMT']['sum'] / accts_past_7d['END_BAL']['mean'],
            np.nan
        )
        accts_past_7d['MAX_CR_RATIO'] = np.where(
            accts_past_7d['END_BAL']['max'] != 0,
            accts_past_7d['CR_AMT']['max'] / accts_past_7d['END_BAL']['max'],
            np.nan
        )

        use_cols = ['NUM_ACCTS', 'CR_UTIL', 'MAX_CR_UTIL', 'TOTAL_DR_RATIO', 'MAX_DR_RATIO',
                    'TOTAL_CR_RATIO', 'MAX_CR_RATIO', 'END_BAL', 'DR_AMT', 'CR_AMT']
        debit_accts = accts_past_7d.loc[accts_past_7d.index.get_level_values(1) == 'DR', use_cols]
        credit_accts = accts_past_7d.loc[accts_past_7d.index.get_level_values(1) == 'CR', use_cols]
        debit_accts.index = debit_accts.index.droplevel(level=1)
        credit_accts.index = credit_accts.index.droplevel(level=1)
        merged = pd.merge(
            debit_accts,
            credit_accts,
            how='outer',
            left_index=True,
            right_index=True,
            suffixes=('_DEB', '_CRE')
        )
        merged['NUM_ACCTS'] = (merged['NUM_ACCTS_DEB'].fillna(0)
                               + merged['NUM_ACCTS_CRE'].fillna(0))
        merged['TOTAL_FLOW'] = (merged['DR_AMT_DEB']['sum'].fillna(0)
                                - merged['CR_AMT_DEB']['sum'].fillna(0)
                                + merged['CR_AMT_CRE']['sum'].fillna(0)
                                - merged['DR_AMT_CRE']['sum']).fillna(0)
        merged['DEB_TO_CRE'] = (merged['DR_AMT_DEB']['sum'].fillna(0)
                                - merged['CR_AMT_DEB']['sum'].fillna(0)) / np.where(
            merged['CR_AMT_CRE']['sum'].fillna(0)
            - merged['DR_AMT_CRE']['sum'].fillna(0) == 0,
            np.nan,
            merged['CR_AMT_CRE']['sum'].fillna(0)
            - merged['DR_AMT_CRE']['sum'].fillna(0)
        )
        merged = merged.drop(columns=['END_BAL_DEB', 'DR_AMT_DEB', 'CR_AMT_DEB', 'END_BAL_CRE',
                                      'DR_AMT_CRE', 'CR_AMT_CRE', 'NUM_ACCTS_DEB', 'NUM_ACCTS_CRE',
                                      'CR_UTIL_DEB', 'MAX_CR_UTIL_DEB'])
        # join customer feature
        merged.columns = merged.columns.droplevel(1)
        merged = (
            cust_past
            .set_index('CUST_ID')
            .join(
                merged,
                how='left',
                validate='1:1'
            )
            .fillna(0)
        )  # TODO: do more careful imputation

        cls.dm.save(merged.reset_index(), snap_dt)

    @classmethod
    def if_success(cls, snap_dt: date) -> bool:
        return cls.dm.count(snap_dt) > 0

class EngagementEventSnap(SnapTableStreamGenerator):
    # each event df @ snap date will record future events between snap_date + 1 to snap_date + 8
    dm = SnapshotDataManagerFileSystem(schema='EVENTS', table='ENGAGE')
    upstreams = [_CurrentStream(EngagementFeatureSnap())]
    
    @classmethod
    def load_model(cls, snap_dt: date):
        model = load_fake_model_by_timetable(
            model_name = 'event',
            snap_dt = snap_dt
        )
        return model

    @classmethod
    def execute(cls, snap_dt: date):
        # get features
        sdp_feat = EngagementFeatureSnap.dm
        features_df = sdp_feat._pd(snap_dt)

        # get model
        model = cls.load_model(snap_dt)

        # predict probability of event for next 7 days
        probs_event_next_7d = model.predict(features_df)

        # convert from 7d prob to 1d prob
        probs_event_daily = 1 - (1 - probs_event_next_7d) ** (1 / 7)

        # construct daily events
        daily_prob_matrix = np.tile(probs_event_daily.values.reshape(-1, 1), 7)
        random_event_matrix = np.random.uniform(
            0,
            1,
            size=(len(probs_event_daily), 7)
        )  # random event value ~ uniform(0,1)
        # argwhere returns a list of (i,j) indices (coordinates) that meet the condition
        # e,g, [(customer idx, on which day event happens)]
        daily_event_matrix = pd.DataFrame(
            np.argwhere(random_event_matrix <= daily_prob_matrix),
            columns=['CUST_IDX', 'EVENT_DAY']
        )
        daily_event_matrix['CUST_ID'] = (
            features_df
            .loc[daily_event_matrix['CUST_IDX'], 'CUST_ID']
            .reset_index(drop = True)
        )
        daily_event_matrix['SNAP_DT'] = snap_dt
        daily_event_matrix['SNAP_DT'] = pd.to_datetime(
                                            daily_event_matrix['SNAP_DT'],
                                            utc = True)
        #daily_event_matrix['EVENT_DT'] = daily_event_matrix['SNAP_DT'] + timedelta(days=2)
        daily_event_matrix['EVENT_TS'] = pd.to_datetime(daily_event_matrix['SNAP_DT'], utc = True) \
                                           + pd.TimedeltaIndex(
                                            daily_event_matrix['EVENT_DAY'] + 1,
                                            unit = 'D') \
                                           + pd.TimedeltaIndex(
                                            np.random.normal(
                                                3600 * 12, 3600 * 6, 
                                                size = len(daily_event_matrix)
                                                ).clip(100, 3600 * 24 - 100).astype('int'),
                                            unit = 's')

        # event channel
        daily_event_matrix['EVENT_CHANNEL'] = np.random.choice(
            ['E', 'P'],
            size=len(daily_event_matrix),
            p=[0.7, 0.3]
        )
        # event type
        daily_event_matrix.loc[
            daily_event_matrix['EVENT_CHANNEL'] == 'E', 'EVENT_CD'
        ] = np.random.choice(
            ['O', 'C'],
            size = len(daily_event_matrix[daily_event_matrix['EVENT_CHANNEL'] == 'E']),
            p = [0.7, 0.3]
        )
        daily_event_matrix.loc[
            daily_event_matrix['EVENT_CHANNEL'] == 'P', 'EVENT_CD'
        ] = np.random.choice(
            ['O', 'C', 'D'],
            size = len(daily_event_matrix[daily_event_matrix['EVENT_CHANNEL'] == 'P']),
            p = [0.5, 0.3, 0.2]
        )

        daily_event_matrix = (
            daily_event_matrix
            .drop(columns=['CUST_IDX', 'EVENT_DAY'])
        )

        cls.dm.save(daily_event_matrix, snap_dt)

    @classmethod
    def if_success(cls, snap_dt: date) -> bool:
        return cls.dm.count(snap_dt) > 0

class EngagementEventFeatureSnap(SnapTableStreamGenerator):
    # features built from past 7d engagement events
    dm = SnapshotDataManagerFileSystem(schema='FEATURES', table='ENGAGE_EVENT_FE')
    upstreams = [
        _PastStream(EngagementEventSnap(), history=14, freq='d')
    ]

    @classmethod
    def execute(cls, snap_dt: date):
        sdp_eng_evnts = EngagementEventSnap.dm
        # events table records event during snap_dt+1 to snap_dt+8
        # so event data for snap_dt-14 contain events from snap_dt-13 to snap_dt-6
        past_14ds = [snap_dt - timedelta(days=d) for d in range(14)]
        eng_evnts_past7d = sdp_eng_evnts.reads_pd(past_14ds)
        eng_evnts_past7d = eng_evnts_past7d[
            eng_evnts_past7d['EVENT_TS'].between(
                dt2str(snap_dt - timedelta(6)),
                dt2str(snap_dt)
            )
        ].drop(columns = ['SNAP_DT']) # take only events during past 7 days
        eng_evnts_past7d['EVENT_DT'] = eng_evnts_past7d['EVENT_TS'].dt.floor('d')
        # make features
        agg_master = (
            eng_evnts_past7d
            .groupby('CUST_ID')
            ['EVENT_DT']
            .agg(['max', 'count'])
            .rename(columns={
                'max' : 'LATEST_EVENT_DT',
                'count' : 'NUM_EVENT'
            })
        )
        agg_master['SNAP_DT'] = snap_dt
        agg_master['SNAP_DT'] = pd.to_datetime(agg_master['SNAP_DT'], utc=True)
        agg_master['DAYS_SINCE_LAST_EVENT'] = (
            agg_master['SNAP_DT'] - agg_master['LATEST_EVENT_DT']
        ).dt.days
        agg_channel = (
            eng_evnts_past7d
            .pivot_table(
                index = 'CUST_ID',
                columns = 'EVENT_CHANNEL',
                values = 'EVENT_DT',
                aggfunc = 'count',
                fill_value = 0,
            ).rename(columns={
                'E' : 'NUM_ENGE_EMAIL',
                'P' : 'NUM_ENGE_PHONE'
            })
        )
        agg_type = (
            eng_evnts_past7d
            .pivot_table(
                index='CUST_ID',
                columns='EVENT_CD',
                values='EVENT_DT',
                aggfunc='count',
                fill_value=0,
            ).rename(columns={
                'C' : 'NUM_CLICK',
                'D' : 'NUM_DECLINE',
                'O' : 'NUM_OPEN'
            })
        )
        agg_master = (
            agg_master
            .drop(columns=['LATEST_EVENT_DT'])
            .join(
                agg_channel,
                how = 'left',
                validate = '1:1'
            ).join(
                agg_type,
                how = 'left',
                validate = '1:1'
            )
        )
        cls.dm.save(agg_master.reset_index(), snap_dt)

    @classmethod
    def if_success(cls, snap_dt: date) -> bool:
        return cls.dm.count(snap_dt) > 0


class ConversionEventSnap(SnapTableStreamGenerator):

    dm = SnapshotDataManagerFileSystem(schema='EVENTS', table='CONVERSION')
    upstreams = [
        _CurrentStream(EngagementFeatureSnap()),
        _CurrentStream(EngagementEventFeatureSnap()),
    ]
    # model param
    NATURAL_CONV_PROB = 0.015

    @classmethod
    def load_event_model(cls, snap_dt: date):
        model = load_fake_model_by_timetable(
            model_name = 'event',
            snap_dt = snap_dt
        )
        return model

    @classmethod
    def load_conv_model(cls, snap_dt: date):
        model = load_fake_model_by_timetable(
            model_name = 'conversion',
            snap_dt = snap_dt
        )
        return model

    @classmethod
    def execute(cls, snap_dt: date):
        # get features
        sdp_eng_feat = EngagementFeatureSnap.dm
        eng_feat = sdp_eng_feat.read_pd(snap_dt)
        sdp_engevet_feat = EngagementEventFeatureSnap.dm
        engevet_feat = sdp_engevet_feat.read_pd(snap_dt)

        # get model
        event_model = cls.load_event_model(snap_dt)
        conv_model = cls.load_conv_model(snap_dt)

        # get probabilities
        event_probs = event_model.predict(eng_feat)
        conv_probs = conv_model.predict(engevet_feat)

        total_probs = pd.merge(
            event_probs.rename('EVENT_PROB'),
            conv_probs.rename('CONV_PROB'),
            how = 'left',
            left_index = True,
            right_index = True,
            validate = '1:1'
        ).fillna(0)
        total_probs['TOTAL_PROB'] = (
                total_probs['EVENT_PROB'] * total_probs['CONV_PROB']
                + (1 - total_probs['EVENT_PROB']) * cls.NATURAL_CONV_PROB
        )

        # convert from 7d prob to 1d prob
        probs_cov_daily = 1 - (1 - total_probs['TOTAL_PROB']) ** (1 / 7)

        # construct daily events
        daily_prob_matrix = np.tile(probs_cov_daily.values.reshape(-1, 1), 7)
        random_event_matrix = np.random.uniform(
            0,
            1,
            size=(len(probs_cov_daily), 7)
        )  # random event value ~ uniform(0,1)
        # argwhere returns a list of (i,j) indices (coordinates) that meet the condition
        # e,g, [(customer idx, on which day event happens)]
        daily_purchase_matrix = pd.DataFrame(
            np.argwhere(random_event_matrix <= daily_prob_matrix),
            columns=['CUST_IDX', 'PUR_DAY']
        )
        daily_purchase_matrix['CUST_ID'] = (
            eng_feat
            .loc[daily_purchase_matrix['CUST_IDX'], 'CUST_ID']
            .reset_index(drop = True)
        )
        daily_purchase_matrix['SNAP_DT'] = snap_dt
        daily_purchase_matrix['SNAP_DT'] = pd.to_datetime(
            daily_purchase_matrix['SNAP_DT'],
            utc=True)
        # daily_event_matrix['EVENT_DT'] = daily_event_matrix['SNAP_DT'] + timedelta(days=2)
        daily_purchase_matrix['PUR_TS'] = pd.to_datetime(daily_purchase_matrix['SNAP_DT'], utc=True) \
                                        + pd.TimedeltaIndex(
                                        daily_purchase_matrix['PUR_DAY'] + 1,
                                        unit='D') \
                                        + pd.TimedeltaIndex(
                                        np.random.normal(
                                            3600 * 12, 3600 * 6, 
                                            size = len(daily_purchase_matrix)
                                        ).clip(100, 3600 * 24 - 100).astype('int'),
                                        unit = 's')

        # purchase amount
        mu, sigma = lognormal2normal(100, 350)
        daily_purchase_matrix['PUR_AMT'] = np.random.lognormal(
            mu,
            sigma,
            size = len(daily_purchase_matrix)
        ).round(2)
        daily_purchase_matrix['PUR_NUM'] = np.random.randint(
            1,
            15,
            size = len(daily_purchase_matrix)
        )

        daily_purchase_matrix = (
            daily_purchase_matrix
            .drop(columns=['CUST_IDX', 'PUR_DAY'])
        )

        cls.dm.save(daily_purchase_matrix, snap_dt)

    @classmethod
    def if_success(cls, snap_dt: date) -> bool:
        return cls.dm.count(snap_dt) > 0 