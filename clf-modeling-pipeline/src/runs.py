import fire
from datetime import date, timedelta
from luntaiDs.CommonTools.utils import str2dt
from luntaiDs.CommonTools.SnapStructure.dependency import SnapTableStreamGenerator
from src.data_layer.dbapi import HYPER_STORAGE

def test_run(*args, **kws):
    print("Test Running ...")
    print(f"args = {args}")
    print(f"kws = {kws}")

def start_optuna_dashboard(ip: str = '0.0.0.0', port: int = 6543):
    """start optuna dashboard for hyper tunning

    :param str ip: target ip for dashboard, defaults to '0.0.0.0'
    :param int port: target port for dashboard, defaults to 6543
    """
    from optuna_dashboard import run_server

    run_server(
        storage = HYPER_STORAGE,
        host = ip,
        port = port
    )
    
def write_schemas_from_js_2_sm(schema_root_file_path: str = 'src/pipeline/schemas'):
    """intialize/write datawarehouse table schemas to schema manager (e.g., backed by mongodb)

    :param str schema_root_file_path: folder for schemas, defaults to 'src/pipeline/schemas'
    """
    from src.data_layer.table_schemas import TableSchema
    
    TableSchema.write_schemas_from_js_2_sm(schema_root_file_path)
    

TASKS = {
    'test_run' : test_run,
    'start_optuna_dashboard' : start_optuna_dashboard,
    'write_schemas_from_js_2_sm' : write_schemas_from_js_2_sm
}

def run(task: str, *args, **kws):
    task_ = TASKS.get(task)
    task_(*args, **kws)
    
if __name__ == '__main__':
    fire.Fire(run)