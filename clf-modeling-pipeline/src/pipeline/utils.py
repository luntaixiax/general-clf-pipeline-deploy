import logging
import ibis
from datetime import date
from luntaiDs.CommonTools.dtyper import DSchema
from luntaiDs.CommonTools.SnapStructure.dependency import SnapTableStreamGenerator
from luntaiDs.CommonTools.SnapStructure.structure import SnapshotDataManagerWarehouseMixin, SnapshotDataManagerFileSystem
from luntaiDs.ProviderTools.clickhouse.snap_struct import SnapshotDataManagerCHSQL
from src.utils.decorators import classproperty
from src.utils.settings import ENTITY_CFG
from src.dao.table_schemas import TableSchema
from src.dao.data_connection import Connection

SnapshotDataManagerCHSQL.setup(
    db_conf = Connection().DWH_CONF
)
SnapshotDataManagerFileSystem.setup(
    fs = Connection().FS_STORAGE,
    root_dir = f"{Connection.DATA_BUCKET}/fake/data",
)
    
class _BaseSnapTableWarehouse(SnapTableStreamGenerator):
    schema: str
    table: str
    
    @classproperty
    def dm(cls) -> SnapshotDataManagerWarehouseMixin:
        return SnapshotDataManagerCHSQL(
            schema = cls.schema,
            table = cls.table,
            snap_dt_key = ENTITY_CFG.dt_key
        )

    @classmethod
    def init_table(cls, snap_dt: date):
        col_schemas: DSchema = TableSchema.read_schema(
            schema = cls.schema,
            table = cls.table
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
    def get_src_dm(cls, schema: str, table: str) -> SnapshotDataManagerFileSystem:
        return SnapshotDataManagerFileSystem(
            schema = schema,
            table = table
        )
    
    @classmethod
    def read(cls, snap_dt: date) -> ibis.expr.types.Table:
        raise NotImplementedError("")
    
    @classmethod
    def transform(cls, snap_dt: date):
        # load table into data wareshouse
        df = cls.read(snap_dt=snap_dt)
        cls.dm.save_ibis(
            df,
            schema = cls.schema,
            table = cls.table,
        )
        

class SnapTableTransfomer(_BaseSnapTableWarehouse):
        
    @classmethod
    def query(cls, snap_dt: date) -> ibis.expr.types.Table:
        raise NotImplementedError("")
    
    @classmethod
    def transform(cls, snap_dt: date):
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