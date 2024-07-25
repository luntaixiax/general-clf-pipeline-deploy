import logging
import mlflow
from sklearn.metrics import roc_auc_score
from luntaiDs.ProviderTools.mlflow.dtyper import ibis_schema_2_mlflow_schema
from src.model_layer.base import HyperMode
from src.model_layer.preprocess_hub import PreprocessParam
from src.model_layer.feature_sel_hub import FSelParam
from src.model_layer.calibrator_hub import CalibParam
from src.model_layer.modeling_hub import ModelPipeParam
from src.dao.model_registry import MlflowCompositePipeline
from src.dao.dbapi import HYPER_STORAGE, MODEL_REGISTRY
from src.services.models.training_data import fetch_train_data

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
    X_train, y_train, X_test, y_test = fetch_train_data(
        data_id = data_id
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
    y_train_pred = y_train_pred.loc[:, y_train_pred.columns.str.startswith('CALIB_')]
    if y_train_pred.shape[1] == 2:
        y_train_pred = y_train_pred['CALIB_CLS1']
    roc_auc_train = roc_auc_score(y_train_pd, y_train_pred)
    logging.info(f"ROC AUC on training set = {roc_auc_train}")

    y_test_pred = cp.score(X_test_pd)
    y_test_pred = y_test_pred.loc[:, y_test_pred.columns.str.startswith('CALIB_')]
    if y_test_pred.shape[1] == 2:
        y_test_pred = y_test_pred['CALIB_CLS1']
    roc_auc_test = roc_auc_score(y_test_pd, y_test_pred)
    logging.info(f"ROC AUC on testing set = {roc_auc_test}")
    
    # register to model registry
    MODEL_REGISTRY.register(
        model_id = model_id,
        data_id = data_id,
        X_train = X_train,
        cp = cp,
        hyper_mode = hyper_mode,
    )