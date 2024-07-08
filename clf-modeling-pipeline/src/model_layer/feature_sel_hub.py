from dataclasses import dataclass
from typing import Any, List, Literal
from functools import partial
from category_encoders import WOEEncoder
from ibis.expr.schema import Schema
from scipy.spatial.distance import squareform, pdist
import numpy as np
from sklearn.base import clone, BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel, f_classif, mutual_info_classif, SelectPercentile, \
    RFECV, RFE, SequentialFeatureSelector
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import AgglomerativeClustering
from category_encoders.ordinal import OrdinalEncoder
from luntaiDs.CommonTools.dtyper import DSchema
from luntaiDs.ModelingTools.FeatureEngineer.transformers import NamedTransformer, MyImputer, \
    SelectThreshold, SelectKBestByCluster, BinaryConverter, \
    MyFeatureSelector, PreSelectSelector
from luntaiDs.ModelingTools.utils.support import make_present_col_selector
from src.dao.table_schemas import TableSchema
from src.model_layer.base import HyperMode

# for numeric features
def distance_1_corr(m):
    # penalize for correlation > 1, make sure this distance is always < 1
    return 1 - np.abs(1 - pdist(m, metric = 'correlation'))
        
class _EdaBackedFselSklearn:
    """build feature selection pipeline backed by EDA result
    """
    GROUP_COL = 'CUST_ID'
    
    def __init__(self, eda_model_id: str):
        self.eda_model_id = eda_model_id
        
    def __call__(self, hyper_mode: HyperMode) -> BaseEstimator:
        """build preprocessing pipeline

        :param HyperMode hyper_mode: hyper tuning mode and params
        :return BaseEstimator: sklearn-compatible feature selector
        """
        from src.dao.dbapi import EDA_FSEL_REGISTRY
        
        if self.eda_model_id not in EDA_FSEL_REGISTRY.get_model_list():
            raise ValueError(f"Given eda_model_id={self.eda_model_id} not found in registry")
        
        selected_cols, descrp = EDA_FSEL_REGISTRY.load_model(
            model_id = self.eda_model_id
        )
        if hyper_mode.hyper_mode:
            # need to pass group column for subsequent group stratified sampling in modeling
            feature_sel_pipe = PreSelectSelector(
                pre_selected_features = [self.GROUP_COL] + selected_cols
            )
        else:
            feature_sel_pipe = PreSelectSelector(pre_selected_features = selected_cols)
        
        return feature_sel_pipe

@dataclass
class CategFselOptions:
    USE_THRESHOLD: bool = True # whether use threshold based or percentile based
    THRESHOLD: float | None = 0.005 # if USE_THRESHOLD = True, use this
    PERCENTILE: int | None = 75 # if USE_THRESHOLD = False, use this
    
@dataclass
class NumFselOptions:
    USE_GROUP_KBEST: bool = False # whether use cluster kbest selector
    P_THRESHOLD: float = 0.05 # if p_value threshold for F-Anova p test
    DENDRO_THRESHOLD: float | None = 0.7 # if USE_GROUP_KBEST = True, need specify this
    TOP_K_CLUSTER: int | None = 2 # max num of features per cluster
    
@dataclass
class FilterMethodOptions:
    num_options: NumFselOptions = NumFselOptions()
    categ_options: CategFselOptions = CategFselOptions()

class _FilterFselSklearn:
    """build feature selection pipeline based on custom fsel pipeline
    and use filter based method - univariate correlation with target
    """
    GROUP_COL = 'CUST_ID'
    
    def __init__(self, method_options: FilterMethodOptions):
        self.method_options = method_options
    
    def __call__(self, hyper_mode: HyperMode) -> BaseEstimator:
        """build preprocessing pipeline

        :param HyperMode hyper_mode: hyper tuning mode and params
        :return BaseEstimator: sklearn-compatible feature selector
        """
        # for binary categ features
        if self.method_options.categ_options.USE_THRESHOLD:
            # use threshold if known
            categ_sel = SelectThreshold(
                score_func = partial(
                    mutual_info_classif, 
                    discrete_features = True
                ), 
                threshold = self.method_options.categ_options.THRESHOLD
            )
        else:
            # use percentile
            categ_sel = SelectPercentile(
                score_func = partial(
                    mutual_info_classif, 
                    discrete_features = True
                ), 
                percentile = self.method_options.categ_options.PERCENTILE
            )
            
        binary_selector = MyFeatureSelector(
            # select features whose mutual info score is higher than threshold
            selector = clone(categ_sel),
            preprocess_pipe = Pipeline([
                # binarize
                ('binarize', BinaryConverter(
                    pos_values = [1, True, 'Y'], 
                    keep_na = True)
                ),
                # imputation
                ('impute', MyImputer(
                    strategy = 'constant', 
                    fill_value = -1, 
                    add_indicator = False
                ))
            ])
        )
        # for categorical features
        categ_selector = MyFeatureSelector(
            selector = clone(categ_sel),
            preprocess_pipe = Pipeline([
                #('int_stringfy', FunctionTransformer(func=intStringfy)),  # convert values into string format
                # ordinal encode categorical varaible, will handle unknown and missing values
                ('ordinal', OrdinalEncoder(
                    handle_unknown = 'value', 
                    handle_missing = 'value')
                ),
            ])
        )

        if self.method_options.num_options.USE_GROUP_KBEST:
            num_sel = SelectKBestByCluster(
                k = self.method_options.num_options.TOP_K_CLUSTER,  # select top 2 within each cluster
                # score is defined using Anova-F
                scorer = SelectThreshold(
                    score_func = f_classif, # use Anova-F as score metric
                    threshold = self.method_options.num_options.P_THRESHOLD,  # 95% significant level
                    use_p = True, # use p-value instead of F score to filter
                ),
                # cluster is defined using agglomerative clustering
                cluster_kernal = AgglomerativeClustering(
                    n_clusters = None, # as we will set distance threshold
                    affinity = 'precomputed', # use precomputed scores
                    linkage = 'complete', # complete linkage
                    distance_threshold = self.method_options.num_options.DENDRO_THRESHOLD, # based on dendrogram above 
                    compute_full_tree = True
                ),
                distance_func = distance_1_corr # defined above
            )
        else:
            num_sel = SelectThreshold(
                score_func = f_classif, # use Anova-F as score metric
                threshold = self.method_options.num_options.P_THRESHOLD,  # 95% significant level
                use_p = True, # use p-value instead of F score to filter
            )
            
        num_selector = MyFeatureSelector(
            # selector criteria
            selector = num_sel,
            preprocess_pipe = Pipeline([
                ('impute', MyImputer(strategy = 'median')),  # convert values into string format
            ])
        )

        # schema
        schema_features: DSchema = TableSchema.read_schema(
            schema = 'FEATURE', 
            table = 'FEATURES'
        )
        ibis_schema: Schema = schema_features.ibis_schema
        
        transformers = []
        if hyper_mode.hyper_mode:
            transformers.append(
                (self.GROUP_COL, 'passthrough', make_present_col_selector([self.GROUP_COL]))
            )
        
        for col, ibis_dtype in ibis_schema.fields.items():
            if col == self.GROUP_COL:
                continue
            
            if ibis_dtype.is_boolean():
                transformers.append(
                    (col, clone(binary_selector), make_present_col_selector([col]))
                )
            elif ibis_dtype.is_numeric():
                transformers.append(
                    (col, clone(num_selector), make_present_col_selector([col]))
                )
            elif ibis_dtype.is_string() and not ibis_dtype.is_date() and not ibis_dtype.is_timestamp():
                transformers.append(
                    (col, clone(categ_selector), make_present_col_selector([col]))
                )
            else:
                continue
                # TODO: need to add them to the list -- find a universal way of preprocessing unknown type columns
        
        pipe = NamedTransformer(
            ColumnTransformer(
                transformers = transformers,
                remainder = 'drop',
                #n_jobs = -1
            )
        )
        return pipe
    
    
@dataclass
class WrapperMethodOptions:
    base_learner: Literal['linear', 'forest'] = 'forest' # if linear will use sgd, otherwise will use random forest
    method: Literal['backward', 'forward', 'RFE', 'RFECV'] = 'backward'
    n_features_to_select: float | None = None # only used for backward/forward/RFE
    min_features_to_select: int | None = None # only used for RFECV
    scoring: str | None = None # only used for backward/forward/RFECV
    
class _WrapperFselSklearn:
    """build feature selection pipeline based on custom fsel pipeline
    and use wrapper based method - recursive or sequential feature selection
    """
    GROUP_COL = 'CUST_ID'
    
    def __init__(self, method_options: WrapperMethodOptions):
        if method_options.method in ('RFE', 'backward', 'forward'):
            assert method_options.n_features_to_select is not None, 'Must provide n_features_to_select for backward/forward/RFE type'
        if method_options.method == 'RFECV':
            assert method_options.min_features_to_select is not None, 'Must provide min_features_to_select for RFECV type'
        
        self.method_options = method_options
        
    def __call__(self, hyper_mode: HyperMode) -> BaseEstimator:
        """build preprocessing pipeline

        :param HyperMode hyper_mode: hyper tuning mode and params
        :return BaseEstimator: sklearn-compatible feature selector
        """
        # construct preprocssing pipeline
        binary_preproc = Pipeline([
            # binarize
            ('binarize', BinaryConverter(
                pos_values = [1, True, 'Y'], 
                keep_na = True)
            ),
            # imputation
            ('impute', MyImputer(
                strategy = 'constant', 
                fill_value = -1, 
                add_indicator = False
            ))
        ])
        categ_preproc = Pipeline([
            #('int_stringfy', FunctionTransformer(func=intStringfy)),  # convert values into string format
            # ordinal encode categorical varaible, will handle unknown and missing values
            ('woe', WOEEncoder(
                    handle_unknown='value', 
                    handle_missing='value',
                    randomized=True, # prevent overfitting
                    sigma=0.05
                )
            ),
        ])
        num_preproc = Pipeline([
            ('impute', MyImputer(strategy = 'median')),
        ])
        

        # classify columns into different types
        schema_features: DSchema = TableSchema.read_schema(
            schema = 'FEATURE', 
            table = 'FEATURES'
        )
        ibis_schema: Schema = schema_features.ibis_schema
        
        transformers = []
        if hyper_mode.hyper_mode:
            transformers.append(
                (self.GROUP_COL, 'passthrough', make_present_col_selector([self.GROUP_COL]))
            )

        for col, ibis_dtype in ibis_schema.fields.items():
            if col == self.GROUP_COL:
                continue
            if ibis_dtype.is_boolean():
                transformers.append(
                    (col, clone(binary_preproc), make_present_col_selector([col]))
                )
            elif ibis_dtype.is_string() and not ibis_dtype.is_date() and not ibis_dtype.is_timestamp():
                transformers.append(
                    (col, clone(categ_preproc), make_present_col_selector([col]))
                )
            elif ibis_dtype.is_numeric():
                transformers.append(
                    (col, clone(num_preproc), make_present_col_selector([col]))
                )
            else:
                continue
                # TODO: need to add them to the list -- find a universal way of preprocessing unknown type columns
        
        preprocess_pipe = NamedTransformer(
            ColumnTransformer(
                transformers = transformers,
                remainder = 'drop',
                #n_jobs = -1
            )
        )
        
        # compile feature selection pipe
        if self.method_options.base_learner == 'forest':
            clf = RandomForestClassifier()
        elif self.method_options.base_learner == 'linear':
            clf = SGDClassifier()
        else:
            raise ValueError("Only support linear/forest type of base leaner")
        
        if self.method_options.method == 'RFECV':
            fsel = RFECV(
                # use random forest as base filter
                estimator = clf,
                min_features_to_select = self.method_options.min_features_to_select,  # keep 50% of features
                scoring = self.method_options.scoring, # we care about precision-recall
                importance_getter = 'auto', # will use feature importance
                cv = 3, # use 3-fold cross validation
                #n_jobs = -1,
                #verbose = 3,
            )
        elif self.method_options.method == 'RFE':
            fsel = RFE(
                estimator = clf,
                n_features_to_select = self.method_options.n_features_to_select
            )
        elif self.method_options.method in ['forward', 'backward']:
            fsel = SequentialFeatureSelector(
                estimator = clf,
                n_features_to_select = self.method_options.n_features_to_select,
                direction = self.method_options.method,
                scoring = self.method_options.scoring
            )
        else:
            raise ValueError("Only support forward/backward/RFE/RFECV methods")
        
        pipe = MyFeatureSelector(
            selector = fsel,
            # need to preprocess feature before feeding into the selector
            preprocess_pipe = preprocess_pipe
        )
        return pipe
                
        
@dataclass
class EmbeddedMethodOptions:
    base_learner: Literal['linear', 'forest'] = 'forest' # if linear will use sgd, otherwise will use random forest
    threshold: str | float | None = None # can be "1.25*mean, 1.3*median", below which will be discarded


class _EmbeddedFselSklearn:
    """build feature selection pipeline based on custom fsel pipeline
    and use Embedded based method - use a meta learner with auto importance getter
    """
    GROUP_COL = 'CUST_ID'
    
    def __init__(self, method_options: EmbeddedMethodOptions):        
        self.method_options = method_options
        
    def __call__(self, hyper_mode: HyperMode) -> BaseEstimator:
        """build preprocessing pipeline

        :param HyperMode hyper_mode: hyper tuning mode and params
        :return BaseEstimator: sklearn-compatible feature selector
        """
        # construct preprocssing pipeline
        binary_preproc = Pipeline([
            # binarize
            ('binarize', BinaryConverter(
                pos_values = [1, True, 'Y'], 
                keep_na = True)
            ),
            # imputation
            ('impute', MyImputer(
                strategy = 'constant', 
                fill_value = -1, 
                add_indicator = False
            ))
        ])
        categ_preproc = Pipeline([
            #('int_stringfy', FunctionTransformer(func=intStringfy)),  # convert values into string format
            # ordinal encode categorical varaible, will handle unknown and missing values
            ('woe', WOEEncoder(
                    handle_unknown='value', 
                    handle_missing='value',
                    randomized=True, # prevent overfitting
                    sigma=0.05
                )
            ),
        ])
        num_preproc = Pipeline([
            ('impute', MyImputer(strategy = 'median')),
        ])
        

        # classify columns into different types
        schema_features: DSchema = TableSchema.read_schema(
            schema = 'FEATURE', 
            table = 'FEATURES'
        )
        ibis_schema: Schema = schema_features.ibis_schema
        
        transformers = []
        if hyper_mode.hyper_mode:
            transformers.append(
                (self.GROUP_COL, 'passthrough', make_present_col_selector([self.GROUP_COL]))
            )

        for col, ibis_dtype in ibis_schema.fields.items():
            if col == self.GROUP_COL:
                continue
            if ibis_dtype.is_boolean():
                transformers.append(
                    (col, clone(binary_preproc), make_present_col_selector([col]))
                )
            elif ibis_dtype.is_string() and not ibis_dtype.is_date() and not ibis_dtype.is_timestamp():
                transformers.append(
                    (col, clone(categ_preproc), make_present_col_selector([col]))
                )
            elif ibis_dtype.is_numeric():
                transformers.append(
                    (col, clone(num_preproc), make_present_col_selector([col]))
                )
            else:
                continue
                # TODO: need to add them to the list -- find a universal way of preprocessing unknown type columns
        
        preprocess_pipe = NamedTransformer(
            ColumnTransformer(
                transformers = transformers,
                remainder = 'drop',
                #n_jobs = -1
            )
        )
        
        # compile feature selection pipe
        if self.method_options.base_learner == 'forest':
            clf = RandomForestClassifier()
        elif self.method_options.base_learner == 'linear':
            clf = SGDClassifier()
        else:
            raise ValueError("Only support linear/forest type of base leaner")

        fsel = SelectFromModel(
            estimator = clf,
            threshold = self.method_options.threshold,
            prefit = False
        )
        
        pipe = MyFeatureSelector(
            selector = fsel,
            # need to preprocess feature before feeding into the selector
            preprocess_pipe = preprocess_pipe
        )
        return pipe


@dataclass
class FSelParam:
    mode: Literal['Free', 'EDA', 'FilterMethod', 'WrapperMethod', 'EmbeddedMethod'] = 'EDA'
    template_id: str | None = None
    eda_model_id: str | None = None
    filter_method_options: FilterMethodOptions | None = None
    wrapper_method_options: WrapperMethodOptions | None = None
    embedded_method_options: EmbeddedMethodOptions | None = None
    
class FSelBaseSklearn:
    FREE_MODE_TEMP_IDS = {
        
    }
    
    def __init__(self, feature_sel_pipe: BaseEstimator):
        self.setPipe(feature_sel_pipe)

    def setPipe(self, feature_sel_pipe: BaseEstimator):
        self.__fsel_pipe = feature_sel_pipe

    def getPipe(self) -> BaseEstimator:
        return self.__fsel_pipe

    @classmethod
    def build(cls, hyper_mode: HyperMode, param: FSelParam) -> BaseEstimator:
        """build feature selection pipeline (static computation graph)

        :param HyperMode hyper_mode: hyper tuning mode and params
        :param FSelParam param: other configurations
        :return BaseEstimator: sklearn feature selector
        """
        if param.mode == 'EDA':
            if param.eda_model_id is None:
                raise ValueError("Must provide eda_model_id if use EDA mode preprocessing")
            
            fsel = _EdaBackedFselSklearn(
                eda_model_id = param.eda_model_id
            )
            
        elif param.mode == 'FilterMethod':
            # filter base method, will need filter_method_options
            if param.filter_method_options == None:
                raise ValueError("Must provide filter_method_options")
            
            fsel = _FilterFselSklearn(
                method_options = param.filter_method_options
            )
            
        elif param.mode == 'WrapperMethod':
            # wrapper base method, will need wrapper_method_options
            if param.wrapper_method_options == None:
                raise ValueError("Must provide wrapper_method_options")
            
            fsel = _WrapperFselSklearn(
                method_options = param.wrapper_method_options
            )
            
        elif param.mode == 'EmbeddedMethod':
            # wrapper base method, will need embedded_method_options
            if param.embedded_method_options == None:
                raise ValueError("Must provide embedded_method_options")
            
            fsel = _EmbeddedFselSklearn(
                method_options = param.embedded_method_options
            )
            
        else:
            # free mode
            if param.template_id is None:
                raise ValueError("Must provide template_id if use Free mode preprocessing")
            if param.template_id not in cls.FREE_MODE_TEMP_IDS:
                raise ValueError(f"{param.template_id} not found, please registry first to code")
            
            fsel = cls.FREE_MODE_TEMP_IDS.get(
                param.template_id
            )() # free mode will not have ability to pass parameter
        
        pipe = fsel(hyper_mode)
        return pipe