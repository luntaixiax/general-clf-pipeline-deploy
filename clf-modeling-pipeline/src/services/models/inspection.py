# model parameter and config
# model performance
# model comparison
# model explain (shap)
from typing import Any, Dict, List, Literal, Tuple
import pandas as pd
import shap
import io
from PIL.Image import Image
from luntaiDs.ModelingTools.Evaluation.metrics import SKClfMetricsEnum, SKMultiClfMetricsCalculator
from src.services.models.registry import load_cp_model, load_mlflow_model, load_shap_explainer
from src.services.models.training_data import fetch_train_data
from src.utils.settings import TARGET_CFG

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

def get_metric_calculators(model_ids: List[str], data_id: str, use_train: bool = False) \
        -> Dict[str, Tuple[SKMultiClfMetricsCalculator, SKMultiClfMetricsCalculator]]:
    """get metric calculator for multi-class sklearn model

    :param List[str] model_ids: the model ids registered to be evaluated/compared
    :param str data_id: the data id  from data registry to be load for testing
    :param bool use_train: whether to evaluate on train or test part
    :return Dict[str, Tuple[SKMultiClfMetricsCalculator, SKMultiClfMetricsCalculator]]: 
            for each model, the score calculators to ease usage of score, roc_auc_curve, etc.
            1st element is for calibrated, 2nd element is for uncalibrated
            return is {model_id: [calib_scorer, uncalib_scorer]}
    """
    
    train_ds, test_ds = fetch_train_data(data_id=data_id)
    # get actual / pred data
    if use_train:
        ds = train_ds.to_pandas()
        X = ds.drop(columns = [TARGET_CFG.target_key])
        y_true = ds[TARGET_CFG.target_key]
    else:
        ds = test_ds.to_pandas()
        X = ds.drop(columns = [TARGET_CFG.target_key])
        y_true = ds[TARGET_CFG.target_key]
    
    # get models
    r = dict()
    for model_id in model_ids:
        # load model
        model = load_cp_model(model_id)
        scorer_calib, scorer_uncalib = model.getMetricCalculator(
            X = X,
            y_true = y_true,
        )
        r[model_id] = (scorer_calib, scorer_uncalib)
    return r
    

def evaluate_models_scores(data_id: str, model_ids: List[str], metrics: List[SKClfMetricsEnum],
        use_train: bool = False) -> Dict[str, Dict[str, Tuple[float, float]]]:
    """get evaluation score metric on given data using given models and metrics

    :param str data_id: the data id from data registry to be load for testing
    :param List[str] model_ids: the model ids registered to be evaluated/compared
    :param List[SKClfMetricsEnum] metric: the metrics should use values from this enum
    :param bool use_train: whether to evaluate on train or test part
    :return Dict[str, Dict[str, Tuple[float, float]]]: {metric: {model_id: [calibrated score, uncalibrated score]}}
    """
    r = get_metric_calculators(
        model_ids = model_ids,
        data_id = data_id,
        use_train = use_train
    )
    score_metrics = dict()
    for metric in metrics:
        scores = dict()
        for model_id, (scorer_calib, scorer_uncalib) in r.items():
            scores[model_id] = (scorer_calib.score(metric), scorer_uncalib.score(metric))
        score_metrics[metric.name] = metric
    return score_metrics
        
        
def get_model_roc_curves(data_id: str, model_id: str, use_uncalib: bool = False) \
        -> Tuple[Dict[int, pd.DataFrame], Dict[int, pd.DataFrame]]:
    """get roc curves using the given model and data

    :param str data_id: the data_id from data registry to be load for testing
    :param str model_id: the model_id from model registry to be load for testing
    :param bool use_uncalib: whether to use calibrated or uncalibrated score
    :return Tuple[Dict[int, pd.DataFrame], Dict[int, pd.DataFrame]]: roc curves
        (
            # training set
            {
                cls0 : Dataframe[threshold, tpr, fpr],
                cls1 : Dataframe[threshold, tpr, fpr]
            },
            # testing set
            {
                cls0 : Dataframe[threshold, tpr, fpr],
                cls1 : Dataframe[threshold, tpr, fpr]
            },
        )
    """
    r_train = get_metric_calculators(
        model_ids = [model_id],
        data_id = data_id,
        use_train = True
    )
    r_test = get_metric_calculators(
        model_ids = [model_id],
        data_id = data_id,
        use_train = False
    )
    train_rocs = r_train[model_id][int(use_uncalib)].roc_auc_curves()
    test_rocs = r_test[model_id][int(use_uncalib)].roc_auc_curves()
    
    return train_rocs, test_rocs
    
def get_model_pr_curves(data_id: str, model_id: str, use_uncalib: bool = False) \
        -> Tuple[Dict[int, pd.DataFrame], Dict[int, pd.DataFrame]]:
    """get precision recall curves using the given model and data

    :param str data_id: the data_id from data registry to be load for testing
    :param str model_id: the model_id from model registry to be load for testing
    :param bool use_uncalib: whether to use calibrated or uncalibrated score
    :return Tuple[Dict[int, pd.DataFrame], Dict[int, pd.DataFrame]]: roc curves
        (
            # training set
            {
                cls0 : Dataframe[threshold, precision, recall],
                cls1 : Dataframe[threshold, precision, recall]
            },
            # testing set
            {
                cls0 : Dataframe[threshold, precision, recall],
                cls1 : Dataframe[threshold, precision, recall]
            },
        )
    """
    r_train = get_metric_calculators(
        model_ids = [model_id],
        data_id = data_id,
        use_train = True
    )
    r_test = get_metric_calculators(
        model_ids = [model_id],
        data_id = data_id,
        use_train = False
    )
    train_prs = r_train[model_id][int(use_uncalib)].precision_recall_curves()
    test_prs = r_test[model_id][int(use_uncalib)].precision_recall_curves()
    
    return train_prs, test_prs

def compare_models_roc_curve(data_id: str, model_ids: List[str], use_uncalib: bool = False, 
        agg: Literal['micro', 'macro'] = 'micro') -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
    """compare roc curve between given models on given dataset
        note when comparing, no need to compare by class (multi-class), just use aggregated roc

    :param str data_id: the data_id from data registry to be load for testing
    :param List[str] model_ids: the model ids registered to be evaluated/compared
    :param bool use_uncalib: whether to use calibrated or uncalibrated score
    :param Literal[&#39;micro&#39;, &#39;macro&#39;] agg:whether use micro or macro averaging, defaults to 'micro'
    :return Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]: result of roc curves
        {
            model_id: (train_df[threshold, tpr, fpr], test_df[threshold, tpr, fpr])
        }
    """
    r_train = get_metric_calculators(
        model_ids = model_ids,
        data_id = data_id,
        use_train = True
    )
    r_test = get_metric_calculators(
        model_ids = model_ids,
        data_id = data_id,
        use_train = False
    )
    
    r = dict()
    for model_id in model_ids:
        train_roc = r_train[model_id][int(use_uncalib)].roc_auc_curve_agg(agg = agg)
        test_roc = r_test[model_id][int(use_uncalib)].roc_auc_curve_agg(agg = agg)
        r[model_id] = (train_roc, test_roc)
        
    return r

def compare_models_pr_curve(data_id: str, model_ids: List[str], use_uncalib: bool = False, 
        agg: Literal['micro', 'macro'] = 'micro') -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
    """compare pr curve between given models on given dataset
        note when comparing, no need to compare by class (multi-class), just use aggregated pr

    :param str data_id: the data_id from data registry to be load for testing
    :param List[str] model_ids: the model ids registered to be evaluated/compared
    :param bool use_uncalib: whether to use calibrated or uncalibrated score
    :param Literal[&#39;micro&#39;, &#39;macro&#39;] agg:whether use micro or macro averaging, defaults to 'micro'
    :return Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]: result of pr curves
        {
            model_id: (train_df[threshold, precision, recall], test_df[threshold, precision, recall])
        }
    """
    r_train = get_metric_calculators(
        model_ids = model_ids,
        data_id = data_id,
        use_train = True
    )
    r_test = get_metric_calculators(
        model_ids = model_ids,
        data_id = data_id,
        use_train = False
    )
    
    r = dict()
    for model_id in model_ids:
        train_pr = r_train[model_id][int(use_uncalib)].precision_recall_curve_agg(agg = agg)
        test_pr = r_test[model_id][int(use_uncalib)].precision_recall_curve_agg(agg = agg)
        r[model_id] = (train_pr, test_pr)
        
    return r

def get_binary_metric_by_thresholds(data_id: str, model_id: str, use_uncalib: bool = False) -> Dict[int, Tuple[pd.DataFrame, pd.DataFrame]]:
    """get binary metrics by thresholds

    :param str data_id: the data_id from data registry to be load for testing
    :param str model_id: the model_id from model registry to be load for testing
    :param bool use_uncalib: whether to use calibrated or uncalibrated score
    :return Dict[int, Tuple[pd.DataFrame, pd.DataFrame]]: the binary metrics result
        {
            cls0: (train_df[threshold, tp, fp, tn, fn, tpr, fpr, precision, recall], 
                    test_df[threshold, tp, fp, tn, fn, tpr, fpr, precision, recall])
        }
    """
    r_train = get_metric_calculators(
        model_ids = [model_id],
        data_id = data_id,
        use_train = True
    )[model_id][int(use_uncalib)]
    r_test = get_metric_calculators(
        model_ids = [model_id],
        data_id = data_id,
        use_train = False
    )[model_id][int(use_uncalib)]
    
    r = dict()
    for cls_idx in range(r_train.num_cls):
        binary_metrics_train = r_train.binary_metrics_by_threshold(cls_idx=cls_idx)
        binary_metrics_test = r_test.binary_metrics_by_threshold(cls_idx=cls_idx)

        r[cls_idx] = (binary_metrics_train, binary_metrics_test)
    return r

def get_shap_global_plots(model_id: str) -> Tuple[Image, Image]:
    """load shap global bar and beeswarm chart on the training set

    :param str model_id: model id for the relevant shap explainer
    :return Tuple[Image.Image, Image.Image]: (bar plot, beeswarm plot)
    """
    from src.dao.dbapi import MODEL_REGISTRY
    
    bar_img, beeswar_img = MODEL_REGISTRY.load_shap_global_plots(
        model_id = model_id
    )
    return bar_img, beeswar_img

def get_shap_local_exp(model_id: str, input_dict: Dict[str, Any]) -> shap._explanation.Explanation:
    """get shap local explanation on given data example

    :param str model_id: model id for the relevant shap explainer
    :param Dict[str, Any] input_dict: the 1-sample input dictionary, 
        schema should be same as whole pipeline input (not pre-model)
    :return shap._explanation.Explanation: shap explanation object, 
        can be use for plot
    """
    mlflow_model = load_mlflow_model(model_id=model_id)
    input_schema = mlflow_model.metadata.signature.inputs
    required_cols = input_schema.input_names()
    required_dtypes = input_schema.pandas_types()
    # validate if input dict contain required columns
    missing_cols = set(required_cols).difference(input_dict.keys())
    if len(missing_cols) > 0:
        raise ValueError(f"Missing columns from input_dict: {missing_cols}")
    
    # convert to pandas df
    X = pd.DataFrame.from_records([input_dict])
    # try to convert to target dtype
    try:
        X = X[required_cols].astype(dict(zip(required_cols, required_dtypes)))
    except Exception as e:
        raise TypeError(f"Expected dtypes: \n{input_schema.inputs}, Get: \n{X.dtypes}, \{e}")
    
    # do the prediction
    model = mlflow_model.unwrap_python_model()
    explainer = load_shap_explainer(model_id=model_id)
    
    X_pre_model = model.transformPreModel(X)
    exp = explainer(X_pre_model)[0]
    return exp

def get_shap_local_waterfall_plot(exp: shap._explanation.Explanation):
    waterfall = shap.plots.waterfall(exp, max_display = 50, show = False)
    return waterfall

def get_shap_local_beeswarm_plot(exp: shap._explanation.Explanation):
    waterfall = shap.plots.beeswarm(exp, max_display = 50, show = False)
    return waterfall
    