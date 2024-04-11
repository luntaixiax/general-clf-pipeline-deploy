from typing import List, Literal, Dict
import pandas as pd
import ibis
from luntaiDs.ModelingTools.Explore.engines.ibis import EDAEngineIbis
from luntaiDs.ModelingTools.Explore.summary import BinaryStatAttr, NominalCategStatAttr, \
        OrdinalCategStatAttr, NumericStatAttr, BinaryStatObj, NumericStatObj, \
            NominalCategStatObj, OrdinalCategStatObj
from luntaiDs.ModelingTools.FeatureEngineer.preprocessing import BaseFeaturePreproc, BinaryFeaturePreproc, \
    NumericFeaturePreproc, NominalCategFeaturePreproc, OrdinalCategFeaturePreproc, \
    BinaryFeaturePreprocModel, NominalFeaturePreprocModel, OrdinalFeaturePreprocModel, \
    NumericFeaturePreprocModel, TabularPreprocModel
from src.data_layer.data_connection import Connection
from src.data_layer.training_data import ConvModelingDataRegistry
from src.data_layer.model_registry import EdaProfilingRegistryMongo
from src.data_layer.dbapi import TableSchema, CH_HANDLER

CMDR = ConvModelingDataRegistry(
    handler = CH_HANDLER, 
    schema = 'TARGET', 
    table = 'TRAINING'
)
CMDR.init_table()

EDA_REGISTRY = EdaProfilingRegistryMongo(
    mongo_client = Connection().MONGO,
    db = 'modeling',
    collection = 'registry_eda'
)


class RemoteDataManagerForRegistry:
    # will not load/save data, but only its reference
    def __init__(self, data_id: str, is_train: bool = False):
        self._data_id = data_id
        self._is_train = is_train
        
    @classmethod
    def get_existing_eda_model_ids(cls) -> List[str]:
        return EDA_REGISTRY.get_model_list()
    
    @classmethod
    def get_existing_eda_data_source_config(cls, eda_model_id: str) -> dict:
        model_config = EDA_REGISTRY.get_model_config(eda_model_id)
        return {
            'source' : model_config['data_source'],
            'config' : model_config['data_config']
        }
    
    @classmethod
    def get_existing_data_ids_from_data_registry(cls) -> List[str]:
        return CMDR.get_existing_ids()
    
    def _get_data(self) -> ibis.expr.types.Table:
        train_ds, test_ds = CMDR.fetch(data_id = self._data_id)
        if self._is_train:
            return train_ds
        else:
            return test_ds
    
    def get_data_preview(self) -> pd.DataFrame:
        return self._get_data().limit(10).to_pandas()
    
    def get_columns(self) -> List[str]:
        data_ = self._get_data()
        return data_.columns
    
    def guess_col_dtype(self, column: str) -> Literal['Binary', 'Ordinal', 'Nominal', 'Numeric']:
        data_ = self._get_data()
        ibis_schema = data_.schema()
        num_unique_values = data_[column].value_counts().count().to_pandas()
        if ibis_schema[column].is_boolean() or num_unique_values == 2:
            return 'Binary'
        if ibis_schema[column].is_string():
            return 'Nominal'
        if ibis_schema[column].is_numeric():
            if num_unique_values < 5:
                return 'Ordinal'
            else:
                return 'Numeric'
            
    def get_value_list(self, column: str) -> list:
        data_ = self._get_data()
        return data_[column].value_counts()[column].to_pandas().tolist()
    
    def train_binary_stat(self, column: str, attr: BinaryStatAttr) -> BinaryStatObj:
        eda = EDAEngineIbis(self._get_data())
        return eda.fit_binary_obj(
            colname = column,
            attr = attr
        )
        
    def train_nominal_stat(self, column: str, attr: NominalCategStatAttr) -> NominalCategStatObj:
        eda = EDAEngineIbis(self._get_data())
        return eda.fit_nominal_obj(
            colname = column,
            attr = attr
        )
    
    def train_ordinal_stat(self, column: str, attr: OrdinalCategStatAttr) -> OrdinalCategStatObj:
        eda = EDAEngineIbis(self._get_data())
        return eda.fit_ordinal_obj(
            colname = column,
            attr = attr
        )
    
    def train_numeric_stat(self, column: str, attr: NumericStatAttr) -> NumericStatObj:
        eda = EDAEngineIbis(self._get_data())
        return eda.fit_numeric_obj(
            colname = column,
            attr = attr
        )
        
    def train_and_save_eda_model(self, eda_model_id: str, serialized_profile: dict) -> list:
        tpm = self.train_eda_model(serialized_profile)
        
        if eda_model_id in EDA_REGISTRY.get_model_list():
            EDA_REGISTRY.remove(model_id = eda_model_id)
        
        EDA_REGISTRY.register(
            model_id = eda_model_id,
            tpm = tpm,
            data_source = 'data_registry',
            data_config = {
                'data_id' : self._data_id,
                'is_train' : self._is_train
            }
        )
        
        return tpm.serialize()
        
    def train_eda_model(self, serialized_profile: dict) -> TabularPreprocModel:
        eda = EDAEngineIbis(self._get_data())
        tpm = TabularPreprocModel()
        for col, config in serialized_profile.items():
            if config['dtype'] == 'Binary':
                attr = BinaryStatAttr.deserialize(config['attr'])
                summary_obj = eda.fit_binary_obj(
                    colname = col,
                    attr = attr
                )
                preproc = BinaryFeaturePreproc.deserialize(config['preproc'])
                model = BinaryFeaturePreprocModel(
                    stat_obj = summary_obj,
                    preproc = preproc
                )
            if config['dtype'] == 'Ordinal':
                attr = OrdinalCategStatAttr.deserialize(config['attr'])
                summary_obj = eda.fit_ordinal_obj(
                    colname = col,
                    attr = attr
                )
                preproc = OrdinalCategFeaturePreproc.deserialize(config['preproc'])
                model = OrdinalFeaturePreprocModel(
                    stat_obj = summary_obj,
                    preproc = preproc
                )
            if config['dtype'] == 'Nominal':
                attr = NominalCategStatAttr.deserialize(config['attr'])
                summary_obj = eda.fit_nominal_obj(
                    colname = col,
                    attr = attr
                )
                preproc = NominalCategFeaturePreproc.deserialize(config['preproc'])
                model = NominalFeaturePreprocModel(
                    stat_obj = summary_obj,
                    preproc = preproc
                )
            if config['dtype'] == 'Numeric':
                attr = NumericStatAttr.deserialize(config['attr'])
                summary_obj = eda.fit_numeric_obj(
                    colname = col,
                    attr = attr
                )
                preproc = NumericFeaturePreproc.deserialize(config['preproc'])
                model = NumericFeaturePreprocModel(
                    stat_obj = summary_obj,
                    preproc = preproc
                )
        
            tpm.append(model)
            
        return tpm
    
    @classmethod       
    def get_existing_eda_model_profile(cls, eda_model_id: str) -> list:
        tpm: TabularPreprocModel = EDA_REGISTRY.load_model(eda_model_id)
        return tpm.serialize()
    
    @classmethod
    def delete_eda_model(cls, eda_model_id: str):
        EDA_REGISTRY.remove(model_id = eda_model_id)
        
    
        
class RemoteDataManagerForQuery(RemoteDataManagerForRegistry):
    def __init__(self, query: str):
        self._query = query
        
    def _get_data(self) -> ibis.expr.types.Table:
        return CH_HANDLER.query(sql = self._query)
    
    def train_and_save_eda_model(self, eda_model_id: str, serialized_profile: dict) -> list:
        tpm = self.train_eda_model(serialized_profile)
        
        if eda_model_id in EDA_REGISTRY.get_model_list():
            EDA_REGISTRY.remove(model_id = eda_model_id)
        
        EDA_REGISTRY.register(
            model_id = eda_model_id,
            tpm = tpm,
            data_source = 'ad_hoc_query',
            data_config = {
                'query' : self._query
            }
        )
        
        return tpm.serialize()
        