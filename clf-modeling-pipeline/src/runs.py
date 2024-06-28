import typer
from datetime import date, timedelta
from luntaiDs.CommonTools.utils import str2dt
from luntaiDs.CommonTools.SnapStructure.dependency import SnapTableStreamGenerator
from src.data_layer.dbapi import HYPER_STORAGE

app = typer.Typer(
    name = "clf-modeling-pipeline"
)

@app.command(help="Start Optuna Dashboard")
def start_optuna_dashboard(
    ip: str = typer.Option(default='0.0.0.0', help='target ip for dashboard'), 
    port: int = typer.Option(default=6543, help='target port for dashboard')
):
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

@app.command(help="Write table schemas from local JSON file to schema manager")
def write_schemas_from_js_2_sm(
    schema_root_file_path: str = typer.Option(
        default='src/pipeline/schemas',
        help='folder for schemas'
    )
):
    """intialize/write datawarehouse table schemas to schema manager (e.g., backed by mongodb)

    :param str schema_root_file_path: folder for schemas, defaults to 'src/pipeline/schemas'
    """
    from src.data_layer.table_schemas import TableSchema
    
    TableSchema.write_schemas_from_js_2_sm(schema_root_file_path)
    
if __name__ == '__main__':
    app()