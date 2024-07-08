from datetime import date
import logging
from typing import List, Literal
import mlflow
from sklearn.metrics import roc_auc_score
from luntaiDs.ProviderTools.mlflow.dtyper import ibis_schema_2_mlflow_schema
from src.model_layer.base import HyperMode
from src.model_layer.preprocess_hub import PreprocessParam
from src.model_layer.feature_sel_hub import FSelParam
from src.model_layer.calibrator_hub import CalibParam
from src.model_layer.modeling_hub import ModelPipeParam
from src.dao.model_registry import MlflowCompositePipeline
from src.dao.dbapi import CONV_MODEL_DATA_REGISTRY, HYPER_STORAGE, MODEL_REGISTRY

def create_train_data(data_id: str, use_snap_dts: List[date],
        sample_method: Literal['simple', 'stratify'] = 'stratify',
        split_method: Literal['simple', 'stratify', 'group', 'timeseries'] = 'group',
        sample_frac: float = 0.25, train_size: float = 0.8, random_seed: int = 0):
    """generate train and test set from database

    :param str data_id: the data id for the generated train/test dataset
    :param List[date] use_snap_dts: list of snap dates to use (sample from)
    :param str sample_method: one of 'simple', 'stratify', defaults to 'stratify'
    :param str split_method: one of 'simple', 'stratify', 'group', 'timeseries', defaults to 'group'
    :param float sample_frac: down sampling size, defaults to 0.25
    :param float train_size: train sample size as of total size, defaults to 0.8
    :param int random_seed: control randomness, defaults to 0
    """
    train_ds, test_ds = CONV_MODEL_DATA_REGISTRY.generate(
        use_snap_dts=use_snap_dts,
        sample_method = sample_method,
        split_method = split_method,
        sample_frac = sample_frac,
        train_size = train_size,
        random_seed = random_seed
    )
    CONV_MODEL_DATA_REGISTRY.register(
        data_id = data_id,
        train_ds = train_ds,
        test_ds = test_ds,
        replace = True
    )
    

def train(data_id: str, model_id: str,
        fsel_param: FSelParam, preproc_param: PreprocessParam, 
        model_param: ModelPipeParam, calib_param: CalibParam):
    """train conversion model

    :param str data_id: the data id for the train/test dataset
    :param str model_id: the model id to uniquely identify the model
    :param FSelParam fsel_param: feature selection parameter
    :param PreprocessParam preproc_param: preprocessing parameter
    :param ModelPipeParam model_param: modeling pipeline parameter
    :param CalibParam calib_param: calibration parameter
    """
    # fetch training data
    X_train, y_train, X_test, y_test = CONV_MODEL_DATA_REGISTRY.fetch(
        data_id = data_id, 
        target_col = CONV_MODEL_DATA_REGISTRY.TARGET_COL
    )
    
    X_train_pd = X_train.to_pandas()
    y_train_pd = y_train.to_pandas()
    X_test_pd = X_test.to_pandas()
    y_test_pd = y_test.to_pandas()
    
    # create static pipeline graph
    cp = MlflowCompositePipeline(
        fsel_param = fsel_param,
        preproc_param = preproc_param,
        model_param = model_param,
        calib_param = calib_param
    )
    # label encoding
    le = cp.buildLabelEncoder()
    y_train_pd = le.fit_transform(y_train_pd)
    
    # hyper tuning
    hyper_mode = HyperMode(
        hyper_mode = True,
        hyper_storage = HYPER_STORAGE
    )
    pipe = cp.buildPipe(hyper_mode)
    pipe.fit(X_train_pd, y_train_pd)
    #print(pipe.transform(X_train_pd).describe())
    logging.info("Hyper Tuning Finished!")
    
    # train
    hyper_mode.hyper_mode = False
    pipe = cp.buildPipe(hyper_mode)
    pipe.fit(X_train_pd, y_train_pd)
    logging.info("Training Finished!")
    
    # calibration
    calib = cp.buildCalib(base_pipe = pipe)
    calib.fit(X_train_pd, y_train_pd)
    logging.info("Calibration Finished!")
    
    cp.setLabelEncoder(le)
    cp.setPipe(pipe)
    cp.setCalib(calib)
    
    # get predition and testing
    logging.info("Getting Prediction and Testing")
    y_train_pred = cp.score(X_train_pd)
    y_train_pred = y_train_pred.loc[:, y_train_pred.columns.str.startswith('calib_')]
    if y_train_pred.shape[1] == 2:
        y_train_pred = y_train_pred['calib_class1']
    roc_auc_train = roc_auc_score(y_train_pd, y_train_pred)
    logging.info(f"ROC AUC on training set = {roc_auc_train}")

    y_test_pred = cp.score(X_test_pd)
    y_test_pred = y_test_pred.loc[:, y_test_pred.columns.str.startswith('calib_')]
    if y_test_pred.shape[1] == 2:
        y_test_pred = y_test_pred['calib_class1']
    roc_auc_test = roc_auc_score(y_test_pd, y_test_pred)
    logging.info(f"ROC AUC on testing set = {roc_auc_test}")
    
    # register to model registry
    MODEL_REGISTRY.register(
        model_id = model_id,
        data_id = data_id,
        cp = cp,
        hyper_mode = hyper_mode,
        signature = mlflow.models.ModelSignature(
            inputs = ibis_schema_2_mlflow_schema(X_train.schema()),
            outputs = mlflow.models.infer_signature(
                model_output = cp.score(X_train_pd.sample(1))
            ).outputs
        )
    )