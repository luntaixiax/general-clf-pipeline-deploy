from typing import List
import json
import pandas as pd
from mlflow.pyfunc import scoring_server, PythonModel
from src.utils.cache import timed_lru_cache
from src.dao.dbapi import MODEL_REGISTRY

@timed_lru_cache(seconds=3600, maxsize=128)
def _load_model(model_id: str) -> PythonModel:
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
    
def predict_batch(model_id: str, df: pd.DataFrame) -> pd.DataFrame:
    """batch prediction using trained model

    :param str model_id: model id of the trained model
    :param pd.DataFrame df: input pandas dataframe
    :return pd.DataFrame: prediction df, of columns 
        [prob_class0, prob_class1,..., calib_class0, calib_class1, ...]
    """
    loaded_model = _load_model(
        model_id = model_id
    )
    return loaded_model.predict(df)


def predict_online(model_id: str, X: List[dict]) -> List[dict]:
    """online serving using trained model

    :param str model_id: model id of the trained model
    :param List[dict] X: list of input feature dictionaries
    :return List[dict]: list of predict outcomes, each of columns 
        [prob_class0, prob_class1,..., calib_class0, calib_class1, ...]
    """
    
    loaded_model = _load_model(
        model_id = model_id
    )
    
    df = pd.DataFrame.from_records(data = X)
    req_dict = {
        'dataframe_split': df.to_dict(orient="split")
    }
    result = scoring_server.invocations(
        data = req_dict,
        content_type = 'application/json',
        model = loaded_model,
        input_schema = loaded_model.metadata.get_input_schema()
    )
    if result.status == 200:
        return json.loads(result.response)['predictions']
    else:
        raise ValueError("Something wrong")