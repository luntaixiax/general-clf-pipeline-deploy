from typing import Generator
import json
import pathlib
from src.data_layer.data_connection import Connection
from luntaiDs.CommonTools.schema_manager import BaseSchemaManager
from luntaiDs.CommonTools.dtyper import DSchema
from luntaiDs.ProviderTools.mongo.schema_manager import MongoSchemaManager

class TableSchema:
    """collection of methods for writing/reading table schema
    class format to easy subclass to work with other type of schema manager
    """
    @classmethod
    def get_schema_manager(cls) -> BaseSchemaManager:
        return MongoSchemaManager(
            mongo_client = Connection().MONGO, 
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