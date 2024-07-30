from typing import List
import json
import pandas as pd
from mlflow.pyfunc import scoring_server
from src.services.models.registry import load_mlflow_model
    
def predict_batch(model_id: str, df: pd.DataFrame) -> pd.DataFrame:
    """batch prediction using trained model

    :param str model_id: model id of the trained model
    :param pd.DataFrame df: input pandas dataframe
    :return pd.DataFrame: prediction df, of columns 
        [PROB_CLS0, PROB_CLS1,..., CALIB_CLS0, CALIB_CLS1, ...]
    """
    loaded_model = load_mlflow_model(
        model_id = model_id
    )
    return loaded_model.predict(df)


def predict_online(model_id: str, X: List[dict]) -> List[dict]:
    """online serving using trained model

    :param str model_id: model id of the trained model
    :param List[dict] X: list of input feature dictionaries
    :return List[dict]: list of predict outcomes, each of columns 
        [PROB_CLS0, PROB_CLS1,..., CALIB_CLS0, CALIB_CLS1, ...]
    """
    
    loaded_model = load_mlflow_model(
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