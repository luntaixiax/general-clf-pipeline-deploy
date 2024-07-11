from typing import Tuple, Dict, Any, Union, Callable, Generator
import logging
import ibis
from datetime import date
from luntaiDs.CommonTools.utils import render_sql_from_file
from luntaiDs.CommonTools.dtyper import DSchema
from luntaiDs.CommonTools.SnapStructure.dependency import SnapTableStreamGenerator
from luntaiDs.ProviderTools.clickhouse.snap_struct import SnapshotDataManagerCHSQL
import pandas as pd
from src.dao.table_schemas import TableSchema
from src.dao.data_connection import Connection

SnapshotDataManagerCHSQL.setup(db_conf = Connection().CH_CONF)

class _BaseSnapTableWarehouse(SnapTableStreamGenerator):
    dm: SnapshotDataManagerCHSQL = None

    @classmethod
    def init_table(cls, snap_dt: date):
        col_schemas: DSchema = TableSchema.read_schema(
            schema = cls.dm.schema,
            table = cls.dm.table
        )
        cls.dm.init_table(
            col_schemas = col_schemas,
            overwrite = False,
            storage_policy = 's3_main'
        )
        
    @classmethod
    def post_check(cls, snap_dt: date):
        logging.info("Void Post Check, will do nothing")
    
    @classmethod
    def transform(cls, snap_dt: date):
        raise NotImplementedError("")
        
    @classmethod
    def execute(cls, snap_dt: date):
        cls.init_table(snap_dt=snap_dt)
        cls.transform(snap_dt=snap_dt)
        cls.post_check(snap_dt=snap_dt)

class SnapTableIngestor(_BaseSnapTableWarehouse):
    
    @classmethod
    def read(cls, snap_dt: date) -> pd.DataFrame:
        raise NotImplementedError("")
    
    @classmethod
    def transform(cls, snap_dt: date):
        # load table into data wareshouse
        df = cls.read(snap_dt=snap_dt)
        cls.dm.save_pandas(
            df,
            schema = cls.dm.schema,
            table = cls.dm.table,
        )
        

class SnapTableTransfomer(_BaseSnapTableWarehouse):
    # sql_template: str = None
    # add_sql_args: Dict[str, Union[str, int, float, Callable[[date], Any]]] = {}
        
    @classmethod
    def query(cls, snap_dt: date) -> ibis.expr.types.Table:
        raise NotImplementedError("")
    
    @classmethod
    def transform(cls, snap_dt: date):
        # process additonal sql arguments, in case some have callable values
        # sql_args = dict()
        # for k, v in cls.add_sql_args.items():
        #     if callable(v):
        #         # render at runtime
        #         sql_args[k] = v(snap_dt) # v must be date -> args
        #     else:
        #         sql_args[k] = v
        # # compile the SQL query
        # sql = render_sql_from_file(
        #     cls.sql_template, 
        #     snap_dt = snap_dt,
        #     **sql_args
        # )
        df = cls.query(snap_dt=snap_dt)
        sql = ibis.to_sql(df)
        # run the SQL query insert into select
        cls.dm.save_qry(
            query = sql,
            snap_dt = snap_dt,
            overwrite = False
        )
    
    
        
def static_ibis_arr_max(arr_col: ibis.expr.operations.Array, arr_length: int):
    return ibis.greatest(
        *[arr_col[i] for i in range(arr_length)]
    )
    
def static_ibis_arr_sum(arr_col: ibis.expr.operations.Array, arr_length: int):
    return sum(
        [arr_col[i] for i in range(arr_length)]
    )
    
def static_ibis_arr_avg(arr_col: ibis.expr.operations.Array, arr_length: int):
    return sum(
        [arr_col[i] for i in range(arr_length)]
    ) / arr_length
    
if __name__ == '__main__':
    TableSchema.write_schemas_from_js_2_sm('schemas')