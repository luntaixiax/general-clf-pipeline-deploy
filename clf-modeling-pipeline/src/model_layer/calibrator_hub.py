from dataclasses import dataclass
from typing import Literal
from scipy.special import logit, expit, softmax
from sklearn.base import clone, BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from luntaiDs.ModelingTools.FeatureEngineer.transformers import ClfTransformer

@dataclass
class CVCalibParam:
    method: Literal['sigmoid', 'isotonic']
    logit: bool = True

class CVCalibratorSklearn:
    """build sigmoid calibration
    """
    def __init__(self, calib_param: CVCalibParam, base_pipe: BaseEstimator):
        self.__base_pipe = base_pipe
        self.__calib_param = calib_param
        
    def __call__(self) -> BaseEstimator:
        if self.__calib_param.logit:
            base = ClfTransformer(
                estimator = self.__base_pipe,
                predict_method = 'predict_proba',
                dense_if_binary = False
            )
            base_estimator = Pipeline([
                ('uncalib', base),
                ('logit', FunctionTransformer(
                    func = softmax
                ))
            ])
        else:
            base_estimator = self.__base_pipe
        return CalibratedClassifierCV(
            base_estimator,
            method = self.__calib_param.method,
            cv = 'prefit', # estimator has been fitted already and all data is used for calibration
            #ensemble = False,
        )

@dataclass
class CalibParam:
    mode: Literal['Free', 'CV'] = 'CV'
    template_id: str | None = None
    calib_param: CVCalibParam | None = None
    
    @property
    def use_origin_x(self) -> bool:
        if self.mode == 'CV':
            return True
        return False
        
class CalibratorBaseSklearn:
    TEMPLATE_IDS = {}
    
    def __init__(self, calib_pipe: BaseEstimator):
        self.setPipe(calib_pipe)

    def setPipe(self, calib_pipe: BaseEstimator):
        self.__calib_pipe = calib_pipe

    def getPipe(self) -> BaseEstimator:
        return self.__calib_pipe
    
    @classmethod
    def build(cls, param: CalibParam, base_pipe: BaseEstimator | None = None) -> BaseEstimator:
        """build calibration pipeline (static computation graph)

        :param CalibParam param: other configurations
        :param BaseEstimator base_pipe: base pipeline, used specifically for CV calibrator
        :return BaseEstimator: sklearn calibration model
        """
        if param.mode == 'CV':
            if param.calib_param is None:
                raise ValueError("Must provide calib_param if use CV mode calibration")
            if base_pipe is None:
                raise ValueError("Must provide base_pipe if use CV mode calibration")
            
            
            calib_builder = CVCalibratorSklearn(
                calib_param = param.calib_param,
                base_pipe = base_pipe
            )
        else:
            raise NotImplementedError("") # TODO
        
        calib = calib_builder()
        return calib