import fire
from datetime import date, timedelta
from CommonTools.utils import str2dt
from CommonTools.SnapStructure.dependency import SnapTableStreamGenerator
from src import etl
from src.utils import update_exec_plan

def test_run(*args, **kws):
    print("Test Running ...")
    print(f"args = {args}")
    print(f"kws = {kws}")
    
def update_exec_plan_2_db():
    update_exec_plan()

def run_pipeline(task_name: str, dag_date: date, day_offset: int = 0, **kws):
    snap_dt = str2dt(dag_date) + timedelta(day_offset)
    print(f"Running {task_name}@{snap_dt} in dettached run mode...")
    
    task: SnapTableStreamGenerator = getattr(etl, task_name)
    task.run_detached(
        snap_dt = snap_dt,
        **kws
    )
    print(f"Completed Running {task_name}@{snap_dt}!")


TASKS = {
    'run_pipeline' : run_pipeline,
    'test_run' : test_run,
    'update_exec_plan_2_db' : update_exec_plan_2_db
}

def run(task: str, *args, **kws):
    task_ = TASKS.get(task, run_pipeline)
    task_(*args, **kws)
    
if __name__ == '__main__':
    fire.Fire(run)