from typing import Tuple, Dict, Any, Union, Callable, Generator
import logging
from datetime import date
import pandas as pd
import json
import pathlib
from CommonTools.utils import render_sql_from_file
from CommonTools.dtyper import DSchema
from CommonTools.schema_manager import BaseSchemaManager
from CommonTools.SnapStructure.dependency import SnapTableStreamGenerator, \
    _CurrentStream, _FutureStream, _PastStream
from ProviderTools.clickhouse.snap_struct import SnapshotDataManagerCHSQL
from ProviderTools.mongo.schema_manager import MongoSchemaManager
from data_connection import CH_CONF, MONGO

SnapshotDataManagerCHSQL.setup(db_conf = CH_CONF)

class TableSchema:
    """collection of methods for writing/reading table schema
    class format to easy subclass to work with other type of schema manager
    """
    @classmethod
    def get_schema_manager(cls) -> BaseSchemaManager:
        return MongoSchemaManager(
            mongo_client = MONGO, 
            database = 'table_schema_ch', 
            collection = 'schemas'
        )
        
    @classmethod
    def iter_schemas_from_js(cls, schema_root_path: str = 'schemas') -> Generator[str, str, dict]:
        """iterate and return each table schema one at time

        :param schema_root_path: root folder that have schema-table json files
        :return: tuple of (schema name, table name, column schema of schema)
        """
        for path in pathlib.Path(schema_root_path).iterdir():
            if path.is_dir():
                schema = path.name
                for f in path.iterdir():
                    table = f.stem # filename, excluding suffix
                    with open(f, 'r') as obj:
                        r = json.load(obj)
                    yield schema, table, r
                
    @classmethod        
    def write_schemas_from_js_2_sm(cls, schema_root_path: str = 'schemas'):
        SM = cls.get_schema_manager()
        for schema, table, r in cls.iter_schemas_from_js(schema_root_path):
            SM.write_raw(schema, table, r)
            
    @classmethod
    def read_schema(cls, schema: str, table: str) -> DSchema:
        SM = cls.get_schema_manager()
        return SM.read(schema, table)
    
    
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