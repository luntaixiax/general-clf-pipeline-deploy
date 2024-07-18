from typing import List, Literal, Dict
import ring
import pandas as pd
import ibis
from luntaiDs.ModelingTools.Explore.engines.base import _BaseEDAEngine
from luntaiDs.ModelingTools.Explore.engines.ibis import EDAEngineIbis
from luntaiDs.ModelingTools.Explore.engines.pandas import EDAEnginePandas
from luntaiDs.ModelingTools.Explore.summary import BinaryStatAttr, NominalCategStatAttr, \
        OrdinalCategStatAttr, NumericStatAttr, BinaryStatObj, NumericStatObj, \
            NominalCategStatObj, OrdinalCategStatObj
from luntaiDs.ModelingTools.FeatureEngineer.preprocessing import BinaryFeaturePreproc, \
    NumericFeaturePreproc, NominalCategFeaturePreproc, OrdinalCategFeaturePreproc, \
    BinaryFeaturePreprocModel, NominalFeaturePreprocModel, OrdinalFeaturePreprocModel, \
    NumericFeaturePreprocModel, TabularPreprocModel
from src.dao.data_connection import Connection
from src.dao.dbapi import CH_HANDLER, EDA_PREPROC_REGISTRY, CONV_MODEL_DATA_REGISTRY


OBJ_HANDLE = Connection().FS_STORAGE

class _BaseDataManager:
    DATA_SOURCE: str = None
    
    @classmethod
    def get_existing_eda_model_ids(cls) -> List[str]:
        return EDA_PREPROC_REGISTRY.get_model_list()
    
    @classmethod
    def get_existing_eda_data_source_config(cls, eda_model_id: str) -> dict:
        model_config = EDA_PREPROC_REGISTRY.get_model_config(eda_model_id)
        return {
            'source' : model_config['data_source'],
            'config' : model_config['data_config']
        }
        
    @classmethod
    def get_existing_data_ids_from_data_registry(cls) -> List[str]:
        return CONV_MODEL_DATA_REGISTRY.get_existing_ids()
    
    @classmethod
    def get_existing_buckets_from_obj_storage(cls) -> List[str]:
        return OBJ_HANDLE.list_buckets()['Name'].tolist()
    
    @classmethod       
    def get_existing_eda_model_profile(cls, eda_model_id: str) -> list:
        tpm: TabularPreprocModel = EDA_PREPROC_REGISTRY.load_model(eda_model_id)
        return tpm.serialize()
    
    @classmethod
    def delete_eda_model(cls, eda_model_id: str):
        EDA_PREPROC_REGISTRY.remove(model_id = eda_model_id)
        
    def get_eda_engine(self) -> _BaseEDAEngine:
        raise NotImplementedError("")
    
    def train_binary_stat(self, column: str, attr: BinaryStatAttr) -> BinaryStatObj:
        eda = self.get_eda_engine()
        return eda.fit_binary_obj(
            colname = column,
            attr = attr
        )
        
    def train_nominal_stat(self, column: str, attr: NominalCategStatAttr) -> NominalCategStatObj:
        eda = self.get_eda_engine()
        return eda.fit_nominal_obj(
            colname = column,
            attr = attr
        )
    
    def train_ordinal_stat(self, column: str, attr: OrdinalCategStatAttr) -> OrdinalCategStatObj:
        eda = self.get_eda_engine()
        return eda.fit_ordinal_obj(
            colname = column,
            attr = attr
        )
    
    def train_numeric_stat(self, column: str, attr: NumericStatAttr) -> NumericStatObj:
        eda = self.get_eda_engine()
        return eda.fit_numeric_obj(
            colname = column,
            attr = attr
        )
        
    def train_eda_model(self, serialized_profile: dict) -> TabularPreprocModel:
        eda = self.get_eda_engine()
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
    
    def _get_data_config(self) -> dict:
        # get data config dict for model registry to work
        raise NotImplementedError("")
    
    def train_and_save_eda_model(self, eda_model_id: str, serialized_profile: dict) -> list:
        tpm = self.train_eda_model(serialized_profile)
        
        if eda_model_id in EDA_PREPROC_REGISTRY.get_model_list():
            EDA_PREPROC_REGISTRY.remove(model_id = eda_model_id)
        
        EDA_PREPROC_REGISTRY.register(
            model_id = eda_model_id,
            tpm = tpm,
            data_source = self.DATA_SOURCE,
            data_config = self._get_data_config()
        )
        
        return tpm.serialize()


class _BaseDataManagerIbis(_BaseDataManager):
    def _get_data(self) -> ibis.expr.types.Table:
        raise NotImplementedError("")
    
    def get_eda_engine(self) -> EDAEngineIbis:
        return EDAEngineIbis(self._get_data())
    
    def get_data_preview(self) -> pd.DataFrame:
        return self._get_data().limit(10).to_pandas()
    
    def get_columns(self) -> List[str]:
        data_ = self._get_data()
        return data_.columns
    
    def guess_col_dtype(self, column: str) -> Literal['Binary', 'Ordinal', 'Nominal', 'Numeric']:
        data_ = self._get_data()
        ibis_schema = data_.schema()
        num_unique_values = (
            data_[column]
            .value_counts()
            .count()
            .to_pandas()
        )
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
        return (
            data_[column]
            .value_counts()
            [column]
            .to_pandas()
            .tolist()
        )
    

class DataManagerIbisForRegistry(_BaseDataManagerIbis):
    DATA_SOURCE = 'data_registry'
    
    # will not load/save data, but only its reference
    def __init__(self, data_id: str, is_train: bool = False):
        self._data_id = data_id
        self._is_train = is_train
        
    def _get_data_config(self) -> dict:
        # get data config dict for model registry to work
        return {
            'data_id' : self._data_id,
            'is_train' : self._is_train
        }
        
    def _get_data(self) -> ibis.expr.types.Table:
        train_ds, test_ds = CONV_MODEL_DATA_REGISTRY.fetch(data_id = self._data_id)
        if self._is_train:
            return train_ds
        else:
            return test_ds

        
class DataManagerIbisForQuery(_BaseDataManagerIbis):
    DATA_SOURCE = 'ad_hoc_query'
    
    def __init__(self, query: str):
        self._query = query
        
    def _get_data_config(self) -> dict:
        # get data config dict for model registry to work
        return {
            'query' : self._query
        }
        
    def _get_data(self) -> ibis.expr.types.Table:
        return CH_HANDLER.query(sql = self._query)
    
    
class _BaseDataManagerPandas(_BaseDataManager):
    
    def _get_data(self, columns: list[str] = None, nrows: int = None) -> pd.DataFrame:
        raise NotImplementedError("")
    
    def get_eda_engine(self) -> EDAEnginePandas:
        return EDAEnginePandas(self._get_data())
    
    def get_data_preview(self) -> pd.DataFrame:
        return self._get_data(nrows = 10).head(10)
    
    def get_columns(self) -> List[str]:
        data_ = self._get_data()
        return data_.columns
    
    def guess_col_dtype(self, column: str) -> Literal['Binary', 'Ordinal', 'Nominal', 'Numeric']:
        data_ = self._get_data(columns = [column])
        vector = data_[column]
        if len(vector.dropna().unique()) == 2:
            return 'Binary'
        if pd.api.types.is_string_dtype(vector.dtype):
            return 'Nominal'
        elif pd.api.types.is_numeric_dtype(vector.dtype):
            if len(vector.unique()) < 5:
                return 'Ordinal'
            else:
                return 'Numeric'
        else:
            return 'Nominal'
            
            
    def get_value_list(self, column: str) -> list:
        data_ = self._get_data(columns = [column])
        return (
            data_[column]
            .unique()
            .tolist()
        )
        
class DataManagerPandasForObjStorage(_BaseDataManagerPandas):
    DATA_SOURCE = 'ad_hoc_file'
    _CACHE_DATA = {}
    
    def __init__(self, bucket: str, file_path: str) -> None:
        self._bucket = bucket
        self._file_path = file_path
        self._obj = OBJ_HANDLE
        self._obj.enter_bucket(bucket)
        
    def _get_data_config(self) -> dict:
        # get data config dict for model registry to work
        return {
            'bucket' : self._bucket,
            'file_path': self._file_path
        }
        
    #@ring.lru(maxsize = 64)
    def _get_data(self, columns: list[str] = None, nrows: int = None) -> pd.DataFrame:
        hash_key = str(hash(self._bucket + self._file_path))
        if hash_key in DataManagerPandasForObjStorage._CACHE_DATA:
            return DataManagerPandasForObjStorage._CACHE_DATA[hash_key]

        # else read from scatch
        if self._file_path.endswith('parquet'):
            df = self._obj.read_parquet(
                self._file_path,
                columns = columns
            )
        elif self._file_path.endswith('csv'):
            df = self._obj.read_csv(
                self._file_path,
                usecols = columns,
                nrows = nrows
            )
        else:
            raise TypeError("Only support parquet/csv")
        print(f'Loaded File: s3://{self._bucket}/{self._file_path}')
        
        DataManagerPandasForObjStorage._CACHE_DATA[hash_key] = df
        
        return df
        