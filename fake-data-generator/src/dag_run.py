import typer
from datetime import date, datetime, timedelta
from luntaiDs.CommonTools.utils import str2dt
from luntaiDs.CommonTools.SnapStructure.dependency import SnapTableStreamGenerator
from src import etl
from src.utils import update_exec_plan
from src.registry import register_predefined_fake_models

app = typer.Typer(
    name = "fake_generator"
)

@app.command(help="Generate task DAG tree and save to Airflow variable")
def update_exec_plan_2_db():
    update_exec_plan()

@app.command(help="Register the fake models (event+conversion) to mongodb")
def register_fake_model():
    register_predefined_fake_models('event')
    register_predefined_fake_models('conversion')
    
@app.command(help="General Entry of pipeline steps")
def run_pipeline(
    task_name: str, 
    dag_date: datetime = typer.Option(help="The base DAG run date"), 
    day_offset: int = typer.Option(default=0, help="Days offset to add to base DAG run date")
):
    snap_dt = str2dt(dag_date.date()) + timedelta(day_offset)
    print(f"Running {task_name}@{snap_dt} in dettached run mode...")
    
    task: SnapTableStreamGenerator = getattr(etl, task_name)
    task.run_detached(
        snap_dt = snap_dt,
    )
    print(f"Completed Running {task_name}@{snap_dt}!")

    
if __name__ == '__main__':
    app()