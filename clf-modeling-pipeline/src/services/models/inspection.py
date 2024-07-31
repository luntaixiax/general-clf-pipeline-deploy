# model parameter and config
# model performance
# model comparison
# model explain (shap)
from typing import Tuple
import pandas as pd
from luntaiDs.ModelingTools.Evaluation.metrics import SKClfMetricsEnum, SKMultiClfMetricsCalculator
from src.services.models.registry import load_cp_model
from src.services.models.training_data import fetch_train_data

def get_coeff_impl(model_id: str) -> pd.Series:
    """get model coefficient or feature importance

    :param str model_id: model id to load
    :return pd.Series: the series containing coeff or importance, index being features
    """
    model = load_cp_model(model_id)
    try:
        r = model.getCoeffImp()
    except TypeError as e:
        r = pd.Series()
    return r

def get_metric_calculator(model_id: str, data_id: str, use_train: bool = False,
        use_calibrated_pred: bool = True) -> SKMultiClfMetricsCalculator:
    """get metric calculator for multi-class sklearn model

    :param str model_id: the model registered to be evaluated
    :param str data_id: the data id  from data registry to be load for testing
    :param bool use_train: whether to evaluate on train or test part
    :param bool use_calibrated_pred: whether to use calibrated preds or uncalibrated preds, defaults to True
    :return SKMultiClfMetricsCalculator: the given metric calculated on given X and ground truth y
    """
    
    # get model
    model = load_cp_model(model_id)
    
    X_train, y_train, X_test, y_test = fetch_train_data(data_id=data_id)
    # get actual / pred data
    if use_train:
        X = X_train.to_pandas()
        y_true = y_train.to_pandas()
    else:
        X = X_test.to_pandas()
        y_true = y_test.to_pandas()
        
    scorer = model.getMetricCalculator(
        X = X,
        y_true = y_true,
        use_calibrated_pred = use_calibrated_pred
    )
    return scorer
    

def evaluate_model_score(model_id: str, data_id: str, metric: SKClfMetricsEnum,
        use_train: bool = False) -> Tuple[float, float]:
    """get evaluation score metric on given data using given model and metric 

    :param str model_id: the model registered to be evaluated
    :param str data_id: the data id  from data registry to be load for testing
    :param SKClfMetricsEnum metric: the metrics should use values from this enum
    :param bool use_train: whether to evaluate on train or test part
    :return Tuple[float, float]: [calibrated score, uncalibrated score]
    """
    # get model
    model = load_cp_model(model_id)
    
    X_train, y_train, X_test, y_test = fetch_train_data(data_id=data_id)
    # get actual / pred data
    if use_train:
        X = X_train.to_pandas()
        y_true = y_train.to_pandas()
    else:
        X = X_test.to_pandas()
        y_true = y_test.to_pandas()
        
    # get uncalibrated or calibrated score
    calibrated_score = model.score(
        X = X,
        y_true = y_true,
        metric = metric,
        use_calibrated_pred = True
    )
    
    uncalibrated_score = model.score(
        X = X,
        y_true = y_true,
        metric = metric,
        use_calibrated_pred = False
    )
    return calibrated_score, uncalibrated_score
        
        
    
    