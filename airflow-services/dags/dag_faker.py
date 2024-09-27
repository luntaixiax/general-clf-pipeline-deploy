
import textwrap
from copy import deepcopy
from datetime import datetime, date, timedelta
import json
from kubernetes.client.models import V1ResourceRequirements
# The DAG object; we'll need this to instantiate a DAG
from airflow.models.dag import DAG
from airflow.providers.cncf.kubernetes.operators.kubernetes_pod import (
    KubernetesPodOperator,
)
from airflow.providers.cncf.kubernetes.secret import Secret
from airflow.models import Variable

TREE = Variable.get(
    'faker_task_tree', 
    deserialize_json = True
)
TASK_CONFIG = Variable.get(
    'faker_task_config', 
    deserialize_json = True
)

TASKS = {}


def generate_task(conf: dict, task_name: str, dag_dt: str, lag: int) -> KubernetesPodOperator:
    cnf = conf.get(task_name) # get task specific configs
    
    # add task id
    if lag > 0:
        suffix = f"_LAG{lag}"
    elif lag == 0:
        suffix = ""
    else:
        suffix = f"_LEAD{-lag}"
    task_id = task_name + suffix
    
    # update command
    cmd = cnf['command']
    cmd.append(f"--dag-date={dag_dt}")
    cmd.append(f"--day-offset={-lag}")
    
    # updatee memory and cpu
    memory = cnf.get('mem_limit', 4096)
    cpu = cnf.get('cpus', 1)
    
    if task_id in TASKS:
        return TASKS[task_id]
    
    task = KubernetesPodOperator(
        kubernetes_conn_id="k8s_cluster",
        image="luntaixia/general-clf-faker:latest",
        namespace='general-clf', # same namespace as secret
        #image="hello-world",
        cmds=cmd,
        name="faker-task-airflow",
        task_id=task_id,
        is_delete_operator_pod=True,
        get_logs=True,
        secrets=[
            Secret(
                deploy_type='volume', # volume type or env type
                deploy_target='/mnt/secrets/', # volume binding point
                secret='vault-secret', # Name of the secrets object in Kubernetes
            )
        ],
        env_vars={
            'ENV' : 'prod',
            'SECRET_TOML_PATH' : '/mnt/secrets/secrets.toml' # match with secret mount point
        },
        # container_resources=V1ResourceRequirements(
        #     limits={
        #         'cpu': cpu,
        #         'memory': f"{memory}m"
        #     }
        # )
    )
    # add task into TASKS global var
    TASKS[task_id] = task
    
    return task
    

def generate_task_tree(conf: dict, tree: dict, dag_dt: str) -> KubernetesPodOperator:
    task_name = tree["current"]["task"]
    lag = tree["current"]["lag"] # relative date to run date
    
    # TODO: how to generate the dynamic date -- let task handle
    
    current_tsk = generate_task(deepcopy(conf), task_name, dag_dt, lag)
    upstreams = tree['upstream']
    
    for upstream in upstreams:
        sub_tsk = generate_task_tree(conf, upstream, dag_dt)
        sub_tsk.set_downstream(current_tsk)
    return current_tsk
        
            


with DAG(
    "general_clf_faker_data_generator",
    # These args will get passed on to each operator
    # You can override them on a per-task basis during operator initialization
    default_args={
        "depends_on_past": True,
        "email": ["airflow@example.com"],
        "email_on_failure": False,
        "email_on_retry": False,
        "retries": 3,
        "retry_delay": timedelta(minutes=1),
        # 'queue': 'bash_queue',
        # 'pool': 'backfill',
        # 'priority_weight': 10,
        # 'end_date': datetime(2016, 1, 1),
        # 'wait_for_downstream': False,
        # 'sla': timedelta(hours=2),
        # 'execution_timeout': timedelta(seconds=300),
        # 'on_failure_callback': some_function, # or list of functions
        # 'on_success_callback': some_other_function, # or list of functions
        # 'on_retry_callback': another_function, # or list of functions
        # 'sla_miss_callback': yet_another_function, # or list of functions
        # 'trigger_rule': 'all_success'
    },
    description="fake data generator for general clf pipeline running",
    start_date=datetime(2024, 8, 1),
    catchup=True,
    schedule_interval="00 09 * * *",
    max_active_runs=1,
    max_active_tasks=1, # should not let cust and acct task run in parallel to avoid overwriting with different random runs
    tags=["general_clf", "faker"],
) as dag:
    generate_task_tree(
        conf = TASK_CONFIG,
        tree = TREE,
        dag_dt = "{{ execution_date.strftime('%Y-%m-%d') }}"
    )
