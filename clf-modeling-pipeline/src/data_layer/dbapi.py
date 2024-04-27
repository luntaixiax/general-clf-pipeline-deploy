
import optuna
from luntaiDs.ProviderTools.clickhouse.dbapi import WarehouseHandlerCHSQL
from src.data_layer.data_connection import Connection
from src.data_layer.model_registry import EdaPreprocRegistryMongo, EdaFeatureSelRegistryMongo
from src.data_layer.training_data import ConvModelingDataRegistry

WarehouseHandlerCHSQL.connect(db_conf = Connection().CH_CONF)
CH_HANDLER = WarehouseHandlerCHSQL()

HYPER_STORAGE = optuna.storages.RDBStorage(
    url = Connection().OPTUNA_STORAGE.getConnStr()
)

CONV_MODEL_DATA_REGISTRY = ConvModelingDataRegistry(
    handler = CH_HANDLER, 
    schema = 'TARGET', 
    table = 'TRAINING'
)
CONV_MODEL_DATA_REGISTRY.init_table()

EDA_PREPROC_REGISTRY = EdaPreprocRegistryMongo(
    mongo_client = Connection().MONGO,
    db = 'modeling',
    collection = 'registry_eda_preproc'
)

EDA_FSEL_REGISTRY = EdaFeatureSelRegistryMongo(
    mongo_client = Connection().MONGO,
    db = 'modeling',
    collection = 'registry_eda_fsel'
)
    
