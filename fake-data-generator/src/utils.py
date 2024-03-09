import pymongo
from datetime import date
from ProviderTools.airflow.api import AirflowAPI
from src.etl import ConversionEventSnap
from src.data_connection import MONGO, AIRFLOW_API


def exec_plan(exec, run_dt: date) -> dict:
    up_infos = []
    for up in exec.upstreams:
        up_infos.append(exec_plan(up, run_dt))

    return {
        'current': {
            "task" : exec.obj.__name__, 
            "lag" : (run_dt - exec.snap_dt).days
        },
        'upstream' : up_infos
    }
    
def update_exec_plan():
    random_date = date(2024, 1, 1) # a random date works
    exec = ConversionEventSnap.get_exec_plan(random_date)
    plan = exec_plan(exec, random_date)
    
    # update to mongo
    db = MONGO['airflow']
    collection = db['task_tree']
    
    content = {
        'dag' : 'fake',
        'plan' : plan
    }
    
    collection.replace_one(
        filter = {
            'dag': 'fake',
        }, # find matching record, if any
        replacement = content,
        upsert = True # update if found, insert if not found
    )
    
    # update to airflow
    AIRFLOW_API.upsert_variable(
        key = 'faker_task_tree',
        value = plan
    )