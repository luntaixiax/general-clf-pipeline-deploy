from pendulum import datetime
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.providers.cncf.kubernetes.operators.kubernetes_pod import (
    KubernetesPodOperator,
)
from airflow.providers.cncf.kubernetes.secret import Secret

with DAG(
    dag_id="example_kubernetes_pod",
    schedule="@once",
    start_date=datetime(2024, 8, 30),
) as dag:
    
    example_docker = DockerOperator(
        task_id="task-docker",
        docker_url="tcp://192.168.2.108:2376",
        tls_ca_cert="/opt/airflow/data/ca.pem",
        tls_client_cert="/opt/airflow/data/cert.pem",
        tls_client_key="/opt/airflow/data/key.pem",
        image="hello-world",
        api_version="auto",
        auto_remove="force",
        tty=True,
        
    )

    example_kpo = KubernetesPodOperator(
        kubernetes_conn_id="k8s_cluster",
        image='chef/chef:16.18.29',
        namespace='test', # same namespace as secret
        #image="hello-world",
        cmds=["/bin/sh", "-c", "echo Hello from the Chef container; sleep 3600"],
        name="airflow-test-pod",
        task_id="task-k8s",
        is_delete_operator_pod=True,
        get_logs=True,
        secrets=[
            Secret(
                deploy_type='volume', # volume type or env type
                deploy_target='/mnt/secrets/', # volume binding point
                secret='test-secret', # Name of the secrets object in Kubernetes
            )
        ]
    )

    example_docker >> example_kpo