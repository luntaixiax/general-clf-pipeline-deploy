from typing import Tuple, Dict, Any, Union, Callable, Generator
import logging
from datetime import date
from luntaiDs.CommonTools.utils import render_sql_from_file
from luntaiDs.CommonTools.dtyper import DSchema
from luntaiDs.CommonTools.SnapStructure.dependency import SnapTableStreamGenerator
from luntaiDs.ProviderTools.clickhouse.snap_struct import SnapshotDataManagerCHSQL
from src.data_layer.table_schemas import TableSchema
from src.data_layer.data_connection import Connection

SnapshotDataManagerCHSQL.setup(db_conf = Connection().CH_CONF)
    
class SnapTableCH(SnapTableStreamGenerator):
    dm: SnapshotDataManagerCHSQL = None
    sql_template: str = None
    add_sql_args: Dict[str, Union[str, int, float, Callable[[date], Any]]] = {}
    
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
        # process additonal sql arguments, in case some have callable values
        sql_args = dict()
        for k, v in cls.add_sql_args.items():
            if callable(v):
                # render at runtime
                sql_args[k] = v(snap_dt) # v must be date -> args
            else:
                sql_args[k] = v
        # compile the SQL query
        sql = render_sql_from_file(
            cls.sql_template, 
            snap_dt = snap_dt,
            **sql_args
        )
        # run the SQL query insert into select
        cls.dm.save_qry(
            query = sql,
            snap_dt = snap_dt,
            overwrite = False
        )
    
    @classmethod
    def execute(cls, snap_dt: date):
        cls.init_table(snap_dt=snap_dt)
        cls.transform(snap_dt=snap_dt)
        cls.post_check(snap_dt=snap_dt)
        
        
if __name__ == '__main__':
    TableSchema.write_schemas_from_js_2_sm('schemas')