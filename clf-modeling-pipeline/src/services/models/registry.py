from datetime import datetime, date
from typing import List
import pandas as pd
from mlflow.pyfunc import PythonModel
from src.model_layer.composite import CompositePipeline
from src.dao.dbapi import MODEL_REGISTRY, MODEL_TIMETABLE
from src.utils.decorators import timed_lru_cache

@timed_lru_cache(seconds=3600, maxsize=128)
def load_mlflow_model(model_id: str) -> PythonModel:
    """load model using model id, use timed lru_cache to avoid repetitive loading

    note if model_id is same, the cache will:
        - not reload for 1 hour
        - cache up to 128 different model ids
    :param str model_id: model id of the trained model
    :return PythonModel: the loaded mlflow python model
    """
    return MODEL_REGISTRY.load_mlflow_pyfunc_model(
        model_id = model_id
    )
    
def load_cp_model(model_id: str) -> CompositePipeline:
    """
    :param str model_id: model id of the trained model
    :return CompositePipeline: the loaded composite pipeline python model
    """
    return load_mlflow_model(model_id=model_id).unwrap_python_model()
    
def list_model_ids() -> List[str]:
    return MODEL_REGISTRY.get_model_list()

def get_prod_model_id() -> str:
    return MODEL_REGISTRY.get_prod_model_id()

def delete_model(model_id: str):
    MODEL_REGISTRY.remove(model_id=model_id)
    # clear cache on update any model
    load_mlflow_model.__wrapped__.cache_clear()
    
    
def get_model_config(model_id: str) -> dict:
    return MODEL_REGISTRY.get_model_config(model_id=model_id)

def register_model_to_timetable(model_id: str, effective_dt: datetime, expiry_dt: datetime):
    MODEL_TIMETABLE.register(
        model_id = model_id,
        start = effective_dt,
        end = expiry_dt,
        force = True
    )

def get_model_id_from_timetable(snap_dt: date) -> str:
    return MODEL_TIMETABLE.get_model_id_by_datetime(
        datetime(
            snap_dt.year,
            snap_dt.month,
            snap_dt.day
        )
    )
    
def get_model_timetable(fill_prod: bool = False) -> pd.DataFrame:
    s = MODEL_TIMETABLE.get_schedule()
    if len(s) > 0:
        # s['start'] = s['start'].astype('str')
        # s['end'] = s['end'].astype('str')
        if fill_prod:
            # fill gap periods with prod id
            prod_id = get_prod_model_id()
            for idx in range(len(s) - 1):
                if s.loc[idx, 'end'] < s.loc[idx + 1, 'start']:
                    s = s.append(
                        {
                            'start' : s.loc[idx, 'end'], 
                            'end' : s.loc[idx + 1, 'start'],
                            'model_id' : prod_id
                        },
                        ignore_index = True
                    )

        return s.sort_values(by = 'start', ignore_index=True)
    else:
        return s