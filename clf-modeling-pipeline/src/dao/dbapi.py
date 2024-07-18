
import optuna
from luntaiDs.ProviderTools.clickhouse.dbapi import WarehouseHandlerCHSQL
from luntaiDs.ProviderTools.mongo.serving import ModelTimeTableMongo
from src.dao.data_connection import Connection
from src.dao.model_registry import EdaPreprocRegistryMongo, EdaFeatureSelRegistryMongo, \
        MlflowMongoWholeModelRegistry
from src.dao.data_registry import ConvModelingDataRegistry

WarehouseHandlerCHSQL.connect(db_conf = Connection().DWH_CONF)
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

MODEL_REGISTRY = MlflowMongoWholeModelRegistry(
    mongo_client=Connection().MONGO,
    db = 'modeling',
    collection = 'registry_model'
)

MODEL_TIMETABLE = ModelTimeTableMongo(
    mongo_client = Connection().MONGO,
    db = 'modeling',
    collection = 'timetable_model'
)
