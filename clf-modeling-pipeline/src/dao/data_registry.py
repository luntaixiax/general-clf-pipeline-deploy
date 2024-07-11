import logging
from datetime import date
from typing import List, Literal, Tuple
import ibis
from ibis import _
from luntaiDs.ModelingTools.CustomModel.sampler import SimpleSamplerIbis, StratifiedSamplerIbis
from luntaiDs.ModelingTools.CustomModel.splitter import SimpleSplitterIbis, StratifiedSplitterIbis, \
    GroupSplitterIbis, TimeSeriesSplitterIbis
from luntaiDs.CommonTools.dtyper import DSchema, DSchemaField
from luntaiDs.ProviderTools.clickhouse.dbapi import WarehouseHandlerCHSQL
from luntaiDs.ProviderTools.clickhouse.serving import _BaseModelDataRegistryCH
from src.dao.table_schemas import TableSchema
from src.utils.settings import ENTITY_CFG, TARGET_CFG

class ConvModelingDataRegistry(_BaseModelDataRegistryCH):
    TARGET_COL = TARGET_CFG.target_key
    ENTITY_COL = ENTITY_CFG.entity_key
    TS_COL = ENTITY_CFG.dt_key
    
    def __init__(self, handler: WarehouseHandlerCHSQL, schema: str, table: str):
        """initialize

        :param WarehouseHandlerCHSQL handler: clickhouse handler
        :param str schema: schema for the table storing modeling data
        :param str table: table storing modeling data
        """
        super().__init__(handler, schema, table)
        self.init_table()
        
    def init_table(self):
        """Initialize table structure
        """
        ## step 1. create table schema on the fly
        # adjust schema
        schema_features: DSchema = TableSchema.read_schema(
            schema = 'FEATURE', 
            table = 'FEATURES'
        )
        # add target columns schema
        schema_features[self.TARGET_COL] = DSchemaField(
            dtype = 'Int8',
            args = dict(nullable = False),
            descr = 'binary target event indicator, 0 or 1'
        )
        # add data_id identifier
        schema_features[self.DATA_ID_COL] = DSchemaField(
            dtype = 'String',
            args = dict(nullable = False),
            primary_key = True,
            partition_key = True, # will partition on this column
            descr = 'training data id, use to identify training batch'
        )
        schema_features[self.TRAIN_TEST_IND_COL] = DSchemaField(
            dtype = 'Boolean',
            args = dict(nullable = False),
            partition_key = True, # will partition on this column
            descr = 'indicator for if in training set'
        )
        
        ## step 2. create structure and validate
        # create schema
        self.handler.create_schema(schema = self.schema)
        
        # validate if table already exists
        if self.handler.is_exist(schema = self.schema, table = self.table):
            # if exists, check if schema changes
            existing_schema = self.handler.get_dtypes(
                schema = self.schema, 
                table = self.table
            )
            current_schema = schema_features.ibis_schema
            if not current_schema.equals(existing_schema):
                msg = f"""
                {self.schema}.{self.table} table schema changed:
                Current in DB:
                {existing_schema}
                Now:
                {current_schema}
                
                Recreate table schema or fix it
                """
                raise TypeError(msg)
            return
        
        self.handler.create_table(
            schema = self.schema,
            table = self.table,
            col_schemas = schema_features
        )
        
    def get_joined_X_y(self, use_snap_dts: List[date]) -> ibis.expr.types.Table:
        """get joined feature (X) and target (y) data for given dates

        :param List[date] use_snap_dts: list of snap dates to use (sample from)
        :return ibis.expr.types.Table: joined ibis dataset
        """
        cust_df = (
            self.handler
            .get_table(schema = 'FEATURE', table = 'FEATURES')
            .filter(_[self.TS_COL].isin(use_snap_dts))
        )
        target_df = (
            self.handler
            .get_table(schema = 'TARGET', table = 'PURCHASE_WINDOW')
            .filter(_[self.TS_COL].isin(use_snap_dts))
            # target filter
            .filter(
                (_['NUM_PUR_F7D'] > 1)
                & (
                    (_['TOTAL_PUR_NUM_F7D'] > 5)
                    | (_['TOTAL_PUR_AMT_F7D'] > 50)
                )
            )
            .select(self.ENTITY_COL, self.TS_COL)
        )
        
        dataset = (
            cust_df
            .left_join(
                target_df,
                [
                    cust_df[self.ENTITY_COL] == target_df[self.ENTITY_COL],
                    cust_df[self.TS_COL] == target_df[self.TS_COL]
                ]
            )
            .mutate(
                _[f'{self.ENTITY_COL}_right']
                .isnull()
                .ifelse(0, 1)
                .name(self.TARGET_COL)
            )
            .drop(f'{self.ENTITY_COL}_right', f'{self.TS_COL}_right')
        )
        return dataset
        
    def generate(self, use_snap_dts: List[date], 
                 sample_method: Literal['simple', 'stratify'] = 'stratify',
                 split_method: Literal['simple', 'stratify', 'group', 'timeseries'] = 'group',
                 sample_frac: float = 0.25, train_size: float = 0.8, 
                 random_seed: int = 0
        ) -> Tuple[ibis.expr.types.Table, ibis.expr.types.Table]:
        """generate train and test set from database

        :param List[date] use_snap_dts: list of snap dates to use (sample from)
        :param str sample_method: one of 'simple', 'stratify', defaults to 'stratify'
        :param str split_method: one of 'simple', 'stratify', 'group', 'timeseries', defaults to 'group'
        :param float sample_frac: down sampling size, defaults to 0.25
        :param float train_size: train sample size as of total size, defaults to 0.8
        :param int random_seed: control randomness, defaults to 0
        :return Tuple[ibis.expr.types.Table, ibis.expr.types.Table]: train_ds, test_ds
        """
        ## step 1. merge X and y
        dataset = self.get_joined_X_y(use_snap_dts=use_snap_dts)
        logging.info(f"Preview of joined training dataset:\n {dataset}")
        logging.info(f"Size before sampling down:\n {dataset[self.TARGET_COL].value_counts()}")
        
        ## step 2. sampling down size
        if sample_method == 'simple':
            sampler = SimpleSamplerIbis(
                shuffle_key = self.ENTITY_COL, # shuffle by CUST_ID
                sample_frac = sample_frac,
                random_seed = random_seed,
                verbose = True
            )
        elif sample_method == 'stratify':
            sampler = StratifiedSamplerIbis(
                shuffle_key = self.ENTITY_COL, # shuffle by CUST_ID
                stratify_key = self.TARGET_COL, # stratify sampling by target column
                sample_frac = sample_frac,
                random_seed = random_seed,
                verbose = True
            )
        else:
            raise ValueError("sample_method can only be either simple or stratify")
        
        sampled = sampler.sample(dataset)
        logging.info(f"Size after sampling down:\n {sampled[self.TARGET_COL].value_counts()}")
        
        ## step 2. splitting into train and test set
        if split_method == 'simple':
            splitter = SimpleSplitterIbis(
                shuffle_key = self.ENTITY_COL, # shuffle by CUST_ID
                train_size = train_size,
                random_seed = random_seed,
                verbose = True
            )
        elif split_method == 'stratify':
            splitter = StratifiedSplitterIbis(
                shuffle_key = self.ENTITY_COL, # shuffle by CUST_ID
                stratify_key = self.TARGET_COL, # stratify sampling by target column
                train_size = train_size,
                random_seed = random_seed,
                verbose = True
            )
        elif split_method == 'group':
            splitter = GroupSplitterIbis(
                shuffle_key = self.ENTITY_COL, # shuffle by CUST_ID
                group_key = self.ENTITY_COL, # group by CUST_ID, so each customer id will only appear in either train/test
                train_size = train_size,
                random_seed = random_seed,
                verbose = True
            )
        elif split_method == 'timeseries':
            splitter = TimeSeriesSplitterIbis(
                shuffle_key = self.ENTITY_COL, # shuffle by CUST_ID
                ts_key = self.TS_COL, # split by SNAP_DT, train will be ealier than test
                train_size = train_size,
                random_seed = random_seed,
                verbose = True
            )
        else:
            raise ValueError("split_method can only be either simple/stratify/group/timeseries")
        
        train_ds, test_ds = splitter.split(sampled)
        logging.info(f"Size after splitting to train and test set:\n"
                     f"Training:\n{train_ds[self.TARGET_COL].value_counts()}\n"
                     f"Testing:\n{test_ds[self.TARGET_COL].value_counts()}")
        
        return train_ds, test_ds