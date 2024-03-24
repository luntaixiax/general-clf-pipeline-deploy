import os
import logging
import hvac
import pymongo
import toml
from luntaiDs.ProviderTools.aws.s3 import S3Accessor
from luntaiDs.ProviderTools.clickhouse.dbapi import ClickHouse
from luntaiDs.ProviderTools.airflow.api import AirflowAPI

ENV = os.environ.get("ENV", "prod")
VAULT_MOUNT_POINT = "general_clf"
VAULT_MOUNT_PATH = {
    's3' : f"{ENV}/s3",
    'clickhouse' : f"{ENV}/clickhouse",
    'mongo' : f"{ENV}/mongo",
    'airflow' : f"{ENV}/airflow",
}
SECRET_PATH = os.environ.get("SECRET_TOML_PATH", "../secrets.toml") # vault credentials

def get_section_from_config(config_path: str, section: str) -> dict:
    with open(config_path) as obj:
        CONFIG = toml.load(obj)
    return CONFIG[section]

def get_vault_resp(mount_point: str, path: str) -> dict:
    vault_config = get_section_from_config(SECRET_PATH, section = 'vault')
    client = hvac.Client(
        url = f"http://{vault_config['endpoint']}:{vault_config['port']}",
        token = vault_config['token']
    )
    if client.is_authenticated():
        logging.info("Successfully authenticated by Vault!")
        response = client.secrets.kv.read_secret_version(
            mount_point=mount_point,
            path=path,
            raise_on_deleted_version=True
        )['data']['data']
        return response
    else:
        raise PermissionError("Vault Permission Error")

def get_obj_storage_accessor() -> S3Accessor:
    response = get_vault_resp(
        mount_point = VAULT_MOUNT_POINT,
        path = VAULT_MOUNT_PATH['s3'],
    )
    s3a = S3Accessor(
        aws_access_key_id=response['ACCESS_KEY'],
        aws_secret_access_key=response['SECRET_ACCESS_KEY'],
        endpoint_url=f"http://{response['endpoint']}:{response['port']}",
    )
    return s3a

def get_warehouse_connect() -> ClickHouse:
    response = get_vault_resp(
        mount_point = VAULT_MOUNT_POINT,
        path = VAULT_MOUNT_PATH['clickhouse'],
    )
    
    db_conf = ClickHouse()
    db_conf.bindServer(
        ip = response['endpoint'],
        port = response['port'],
    )
    db_conf.login(
        username = response['username'],
        password = response['password']
    )
    return db_conf

def get_mongo_client() -> pymongo.mongo_client.MongoClient:
    response = get_vault_resp(
        mount_point = VAULT_MOUNT_POINT,
        path = VAULT_MOUNT_PATH['mongo'],
    )
    
    ip = response['endpoint']
    port = response['port']
    username = response['username']
    password = response['password']
    conn_str = f"mongodb://{username}:{password}@{ip}:{port}/"
    return pymongo.MongoClient(conn_str)

def get_airflow_api() -> AirflowAPI:
    response = get_vault_resp(
        mount_point = VAULT_MOUNT_POINT,
        path = VAULT_MOUNT_PATH['airflow'],
    )
    return AirflowAPI(
        host = response['endpoint'],
        port = response['port'],
        username = response['username'],
        password = response['password'],
    )


    
class Singleton (type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class Connection(metaclass=Singleton):
    DATA_BUCKET = "general-clf-pipeline-project"
    
    def __init__(self) -> None:
        self._s3a = get_obj_storage_accessor()
        self._s3a.enter_bucket(self.DATA_BUCKET)
        self._ch_conf = get_warehouse_connect()
        self._mongo = get_mongo_client()
        self._airflow_api = get_airflow_api()
        
    @property
    def S3A(self):
        return self._s3a
    
    @property
    def CH_CONF(self):
        return self._ch_conf
    
    @property
    def MONGO(self):
        return self._mongo
    
    @property
    def AIRFLOW_API(self):
        return self._airflow_api