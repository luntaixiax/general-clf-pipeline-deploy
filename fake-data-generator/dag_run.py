import fire
from datetime import date, timedelta
import importlib
import etl
from CommonTools.utils import str2dt
from CommonTools.SnapStructure.dependency import SnapTableStreamGenerator

def run(task_name: str, dag_date: date, day_offset: int = 0, **kws):
    snap_dt = str2dt(dag_date) + timedelta(day_offset)
    print(f"Running {task_name}@{snap_dt} in dettached run mode...")
    
    task: SnapTableStreamGenerator = getattr(etl, task_name)
    task.run_detached(
        snap_dt = snap_dt,
        **kws
    )
    print(f"Completed Running {task_name}@{snap_dt}!")
    
if __name__ == '__main__': 
    fire.Fire(run)