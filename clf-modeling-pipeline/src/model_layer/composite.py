from dataclasses import asdict
import pandas as pd
from sklearn.base import BaseEstimator, estimator_html_repr
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, FunctionTransformer
import shap
from src.model_layer.base import HyperMode
from src.model_layer.feature_sel_hub import FSelParam, FSelBaseSklearn
from src.model_layer.preprocess_hub import PreprocessParam, PreprocessingBaseSklearn
from src.model_layer.modeling_hub import ModelPipeParam, ModelingBaseSklearn
from src.model_layer.calibrator_hub import CalibParam, CalibratorBaseSklearn

class CompositePipeline:
    def __init__(self, fsel_param: FSelParam, preproc_param: PreprocessParam, 
                 model_param: ModelPipeParam, calib_param: CalibParam):
        self._fsel_param = fsel_param
        self._preproc_param = preproc_param
        self._model_param = model_param
        self._calib_param = calib_param
        
    def getLoggingParams(self) -> dict:
        return {
            'fsel' : asdict(self._fsel_param),
            'preproc' : asdict(self._preproc_param),
            'model' : asdict(self._model_param),
            'calib' : asdict(self._calib_param),
        }
        
    def getLoggingAttrs(self) -> dict:
        model_builder_cls = ModelingBaseSklearn.TEMPLATE_IDS.get(
            self._model_param.template_id
        )
        return {
            'model': model_builder_cls.getLoggingAttrs(
                model = self.getPipe()['model']
            )
        }
        
    
    def buildPipe(self, hyper_mode: HyperMode) -> Pipeline:
        group_col = ModelingBaseSklearn.TEMPLATE_IDS.get(
                self._model_param.template_id
        ).GROUP_COL
        
        def floatfy(X: pd.DataFrame) -> pd.DataFrame:
            columns = X.columns.difference(other = [group_col])
            X[columns] = X[columns].astype('float')
            return X    
        
        fsel_pipe = FSelBaseSklearn.build(
            hyper_mode,
            param = self._fsel_param
        )
        preproc_pipe = PreprocessingBaseSklearn.build(
            hyper_mode = hyper_mode,
            param = self._preproc_param
        )
        model_pipe = ModelingBaseSklearn.build(
            hyper_mode = hyper_mode,
            param = self._model_param
        )
        
        pipeline = Pipeline([
            ('fsel', fsel_pipe),
            ('preproc', preproc_pipe),
            ('floatfy', FunctionTransformer(
                func = floatfy, check_inverse = False
            )),
            ('model', model_pipe)
        ])
        
        return pipeline
    
    def setPipe(self, pipe: Pipeline):
        self.__pipe = pipe
        self.__fsel = FSelBaseSklearn(pipe[0])
        self.__preproc = PreprocessingBaseSklearn(pipe[1])
        self.__modeling = ModelingBaseSklearn(pipe[3])
        
    def getPipe(self) -> Pipeline:
        try:
            return self.__pipe
        except AttributeError:
            raise ValueError(
                "You don't have a model loaded yet, either pass your trained model pipeline to "
                "setPipe() or build a new one by calling .buildPipe() first and load the trained model to setPipe()"
            )
    
    def getFselSK(self) -> FSelBaseSklearn:
        return self.__fsel
    
    def getPreprocSK(self) -> PreprocessingBaseSklearn:
        return self.__preproc
    
    def getModelingSK(self) -> ModelingBaseSklearn:
        return self.__modeling
    
    def getCalibSK(self) -> CalibratorBaseSklearn:
        return self.__calib
    
    def buildCalib(self, base_pipe: BaseEstimator | None = None) -> BaseEstimator:
        return CalibratorBaseSklearn.build(
            param = self._calib_param,
            base_pipe = base_pipe
        )
        
    def setCalib(self, calibrator: BaseEstimator):
        self.__calib_pipe = calibrator
        self.__calib = CalibratorBaseSklearn(calibrator)
        
    def getCalib(self) -> BaseEstimator:
        try:
            return self.__calib_pipe
        except AttributeError:
            raise ValueError(
                "You don't have a calibrator loaded yet, either pass your trained calibrator pipeline to "
                "setCalib() or build a new one by calling .buildCalib() first and load the trained model to setCalib()"
            )
            
    def buildLabelEncoder(self) -> LabelEncoder:
        return LabelEncoder()
    
    def setLabelEncoder(self, le: LabelEncoder):
        self.__le = le
        
    def getLabelEncoder(self) -> LabelEncoder:
        try:
            return self.__le
        except AttributeError:
            raise ValueError(
                "You don't have a LabelEncoder loaded yet, either pass your trained LabelEncoder to "
                "setLabelEncoder() or build a new one by calling .buildLabelEncoder() first "
                "and load the trained one to setLabelEncoder()"
            )
            
    def transformPreModel(self, X: pd.DataFrame) -> pd.DataFrame:
        pre_modeling_pipe = self.getPipe()[:-1]
        X_premodel = pre_modeling_pipe.transform(X)
        return X_premodel
    
    def score(self, X: pd.DataFrame) -> pd.DataFrame:
        df = pd.DataFrame(index = X.index)
        pipe = self.getPipe()
        probs = pipe.predict_proba(X) # (n, n_class)
        
        # get calibrator
        calibrator = self.getCalib()
        if self._calib_param.use_origin_x:
            probs_calib = calibrator.predict_proba(X)
        else:
            probs_calib = calibrator.predict_proba(probs)
        
        # write result
        le = self.getLabelEncoder()
        if probs.shape[1] != len(le.classes_) or probs_calib.shape[1] != len(le.classes_):
            raise ValueError(f"LabelEncoder implies n_class = {len(le.classes_)} while you gave {probs.shape[1]}")
        prob_cols = [f"PROB_CLS{i}" for i in range(len(le.classes_))]
        calib_cols = [f"CALIB_CLS{i}" for i in range(len(le.classes_))]
        df[prob_cols] = probs
        df[calib_cols] = probs_calib
        
        return df
    
    def renderStructureHTML(self) -> str:
        return estimator_html_repr(self.getCalib())
    
    def getShapExplainer(self, X: pd.DataFrame, premodel: bool = False) -> shap.Explainer:
        """get shap explainer for each type of model

        :param pd.DataFrame X: the training data for shap explainer, depend on premodel param
        :param bool premodel: if True, X will be the premodeling X, if False, will be X for whole pipeline
        :return shap.Explainer: the underlying shap explainer
        """
        
        modeling_pipe = self.getModelingSK().getPipe()
        if premodel is False:
            X_pretrain = self.transformPreModel(X)
        else:
            X_pretrain = X
        
        model_builder_cls = ModelingBaseSklearn.TEMPLATE_IDS.get(
            self._model_param.template_id
        )
        return model_builder_cls.get_shap_explainer(
            model = modeling_pipe,
            data = X_pretrain,
        )
        