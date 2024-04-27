from dataclasses import dataclass
from typing import Any, Dict, Literal
from datetime import datetime
import optuna
from sklearn.base import clone, BaseEstimator
from lightgbm import LGBMClassifier
from luntaiDs.ModelingTools.CustomModel.custom import GroupStratifiedOptunaSearchCV
from sklearn.linear_model import SGDClassifier
from src.model_layer.base import HyperMode

@dataclass
class CVParams:
    n_cv: int = 3
    n_trials: int = 30
    scoring: str = 'roc_auc'

@dataclass
class _ModelParam:
    pass

@dataclass
class LGBMHparam(_ModelParam):
    objective: Literal['binary', 'multiclass'] = 'binary'
    n_estimators: int = 100
    boosting_type: Literal['gbdt', 'rf', 'dart', 'goss'] = 'gbdt'
    class_weight: Literal['balanced'] | None = None # balanced


class _BaseModel:
    GROUP_COL = 'CUST_ID'
    
    def __init__(self, cv_params: CVParams, model_params: _ModelParam):
        self._cv_params = cv_params
        self._model_params = model_params
        
    def _get_studyname_prefix(self) -> str:
        """get prefix of study name, will be append with datetime of run

        :return str: study name prefix
        """
        raise NotImplementedError("")
        
    def _get_base_model(self, selected_hparams: dict | None = None) -> BaseEstimator:
        """get model instance, taking self._model_params and selected_hparams as input

        :param dict | None selected_hparams: if not given, will return base model
        :return BaseEstimator: model instance
        """
        raise NotImplementedError("")
        
    def _get_hparam_distr(self) -> Dict[str, optuna.distributions.BaseDistribution]:
        """get optuna hyperparameter tuning param distribution dictionary

        :return Dict[str, optuna.distributions.BaseDistribution]: optuna distribution
            e.g., {'learning_rate': optuna.distributions.FloatDistribution(0.001, 0.35, log=True)}
        """
        raise NotImplementedError("")
        
    def __call__(self, hyper_mode: HyperMode) -> BaseEstimator:
        """build modeling pipeline

        :param HyperMode hyper_mode: hyper tuning mode and params
        :return BaseEstimator: sklearn-compatible estimator
        """
        if hyper_mode.hyper_mode:
            study_name = f"{self._get_studyname_prefix()}-{datetime.now()}"
            # record to attr to pass to non-hyper mode
            hyper_mode.attrs = {
                'study_name' : study_name
            }
            # create study
            study = optuna.create_study(
                storage = hyper_mode.hyper_storage,
                study_name = study_name,
                pruner = optuna.pruners.MedianPruner(),
                direction = 'maximize',
                sampler = optuna.samplers.TPESampler()
            )
            model = GroupStratifiedOptunaSearchCV(
                estimator = self._get_base_model({}),
                group_col = self.GROUP_COL,
                param_distributions = self._get_hparam_distr(),
                n_cv = self._cv_params.n_cv,
                error_score = 'raise',
                n_trials = self._cv_params.n_trials,
                refit = True,
                scoring = self._cv_params.scoring,
                study = study,
                verbose = 3
            )
            
        else:
            # load back study from attrs
            study = optuna.load_study(
                study_name = hyper_mode.attrs['study_name'], 
                storage = hyper_mode.hyper_storage
            )
            model = self._get_base_model(
                selected_hparams = study.best_params
            )
            
        return model
        

class _SingleLayerLGBM(_BaseModel):
    """build modeling pipeline (single layer)
    """
    GROUP_COL = 'CUST_ID'
    
    def __init__(self, cv_params: CVParams, model_params: LGBMHparam):
        super().__init__(cv_params, model_params)
        
    def _get_studyname_prefix(self) -> str:
        """get prefix of study name, will be append with datetime of run

        :return str: study name prefix
        """
        return "lgbm-single"
        
    def _get_base_model(self, selected_hparams: dict | None = None) -> BaseEstimator:
        """get model instance, taking self._model_params and selected_hparams as input

        :param dict | None selected_hparams: if not given, will return base model
        :return BaseEstimator: model instance
        """
        return LGBMClassifier(
            objective = self._model_params.objective, 
            n_estimators = self._model_params.n_estimators, 
            boosting_type = self._model_params.boosting_type, 
            importance_type = 'split',
            class_weight = self._model_params.class_weight,
            **selected_hparams
        )
        
    def _get_hparam_distr(self) -> Dict[str, optuna.distributions.BaseDistribution]:
        """get optuna hyperparameter tuning param distribution dictionary

        :return Dict[str, optuna.distributions.BaseDistribution]: optuna distribution
            e.g., {'learning_rate': optuna.distributions.FloatDistribution(0.001, 0.35, log=True)}
        """
        return {
            # boosting regularization
            'learning_rate': optuna.distributions.FloatDistribution(0.001, 0.35, log=True),
            'reg_alpha': optuna.distributions.FloatDistribution(1e-6, 1e2, log=True),  # l1
            'reg_lambda': optuna.distributions.FloatDistribution(1e-6, 1e2, log=True),  # l2
            # tree regularization
            'max_depth': optuna.distributions.IntDistribution(3, 12, step=3),
            'num_leaves': optuna.distributions.IntDistribution(20, 100, step=10),
            'min_split_gain': optuna.distributions.FloatDistribution(1e-3, 1e2, log=True),  # min_split_loss
            'subsample': optuna.distributions.FloatDistribution(0.3, 1, step=0.1),
            'min_child_weight': optuna.distributions.IntDistribution(1, 20),
            'colsample_bytree': optuna.distributions.FloatDistribution(0.3, 1, step=0.1),
        }
        
        

    
@dataclass
class SGDHparam(_ModelParam):
    loss: Literal['hinge', 'log', 'modified_huber', 
                  'squared_hinge', 'perceptron', 'squared_error', 
                  'huber', 'epsilon_insensitive', 
                  'squared_epsilon_insensitive'] = "hinge"
    fit_intercept: bool = True,
    max_iter: int = 1000,
    early_stopping: bool = False,
    class_weight: Literal['balanced'] | None = None,
    average: int | bool = False
    
    
class _SingleLayerSGD(_BaseModel):
    """build modeling pipeline (single layer)
    """
    GROUP_COL = 'CUST_ID'
    
    def __init__(self, cv_params: CVParams, model_params: SGDHparam):
        super().__init__(cv_params, model_params)
        
    def _get_studyname_prefix(self) -> str:
        """get prefix of study name, will be append with datetime of run

        :return str: study name prefix
        """
        return "sgd-single"
        
    def _get_base_model(self, selected_hparams: dict | None = None) -> BaseEstimator:
        """get model instance, taking self._model_params and selected_hparams as input

        :param dict | None selected_hparams: if not given, will return base model
        :return BaseEstimator: model instance
        """
        return SGDClassifier(
            loss = self._model_params.loss,
            fit_intercept = self._model_params.fit_intercept,
            max_iter = self._model_params.max_iter,
            early_stopping = self._model_params.early_stopping,
            class_weight = self._model_params.class_weight,
            average = self._model_params.average,
            **selected_hparams
        )
        
    def _get_hparam_distr(self) -> Dict[str, optuna.distributions.BaseDistribution]:
        """get optuna hyperparameter tuning param distribution dictionary

        :return Dict[str, optuna.distributions.BaseDistribution]: optuna distribution
            e.g., {'learning_rate': optuna.distributions.FloatDistribution(0.001, 0.35, log=True)}
        """
        base_params =  {
            # regularization
            'penalty': optuna.distributions.CategoricalDistribution(['elasticnet', None]),
            'l1_ratio': optuna.distributions.FloatDistribution(0, 1, log=False),  # l1
            'alpha': optuna.distributions.FloatDistribution(1e-7, 1e7, log=True),  # strength of regularization
            # gradient descent
            'learning_rate': optuna.distributions.CategoricalDistribution(['constant', 'optimal', 'invscaling', 'adaptive']),
            'eta0': optuna.distributions.FloatDistribution(1e-4, 1e4, log=True), # initial learning rate
            'power_t': optuna.distributions.FloatDistribution(1e-2, 1e2, log=True),  # exponent for inverse scaling
        }
        if self._model_params.early_stopping:
            base_params['n_iter_no_change'] = optuna.distributions.IntDistribution(3, 20)
            
        return base_params




    
@dataclass
class ModelPipeParam:
    cv_params: CVParams
    model_params: _ModelParam
    template_id: str | None = None
    
class ModelingBaseSklearn:
    TEMPLATE_IDS = {
        'STANDARD_LGBM' : _SingleLayerLGBM,
        'STANDARD_SGD' : _SingleLayerSGD,
    }
    
    def __init__(self, model_pipe: BaseEstimator):
        self.setPipe(model_pipe)

    def setPipe(self, model_pipe: BaseEstimator):
        self.__model_pipe = model_pipe

    def getPipe(self) -> BaseEstimator:
        return self.__model_pipe

    @classmethod
    def build(cls, hyper_mode: HyperMode, param: ModelPipeParam) -> BaseEstimator:
        """build modeling pipeline (static computation graph)

        :param HyperMode hyper_mode: hyper tuning mode and params
        :param ModelPipeParam param: other configurations
        :return BaseEstimator: sklearn feature model
        """
        if param.template_id is None:
            raise ValueError("Must provide template_id if use Free mode modeling")
        if param.template_id not in cls.TEMPLATE_IDS:
            raise ValueError(f"{param.template_id} not found, please registry first to code")
        
        pipe_cls = cls.TEMPLATE_IDS.get(param.template_id)
        pipe_builder = pipe_cls(
            cv_params = param.cv_params,
            model_params = param.model_params
        )
        pipe = pipe_builder(hyper_mode)
        return pipe
        