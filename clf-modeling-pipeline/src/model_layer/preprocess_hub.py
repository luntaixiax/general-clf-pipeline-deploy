from dataclasses import dataclass
from typing import Any, Literal
from ibis.expr.schema import Schema
from sklearn.base import TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from luntaiDs.CommonTools.dtyper import DSchema
from luntaiDs.ModelingTools.FeatureEngineer.transformers import NamedTransformer, \
    OutlierClipper, MyImputer, BinaryConverter, BucketCategValue
#from luntaiDs.ModelingTools.FeatureEngineer.preprocessing import intStringfy
from luntaiDs.ModelingTools.utils.support import make_present_col_selector
from src.dao.table_schemas import TableSchema
from src.model_layer.base import HyperMode
from src.utils.settings import ENTITY_CFG


class _EdaBasedPreprocessingSklearn:
    """build preprocessing pipeline based on EDA result
    """
    GROUP_COL = ENTITY_CFG.entity_key
    
    def __init__(self, eda_model_id: str):
        self.eda_model_id = eda_model_id
        
    def __call__(self, hyper_mode: HyperMode) -> TransformerMixin:
        """build preprocessing pipeline

        :param HyperMode hyper_mode: hyper tuning mode and params
        :return TransformerMixin: sklearn-compatible transformer
        """
        from src.dao.dbapi import EDA_PREPROC_REGISTRY
        
        if self.eda_model_id not in EDA_PREPROC_REGISTRY.get_model_list():
            raise ValueError(f"Given eda_model_id={self.eda_model_id} not found in registry")
        
        preproc_pipeline: NamedTransformer = (
            EDA_PREPROC_REGISTRY
            .load_model(model_id=self.eda_model_id) # TabularProcessingModel
            .compile_sklearn_pipeline()
        )
        if hyper_mode.hyper_mode:
            # need to pass group column for subsequent group stratified sampling in modeling
            transformers = preproc_pipeline.transformer.transformers
            transformers.append(
                (self.GROUP_COL, 'passthrough', make_present_col_selector([self.GROUP_COL]))
            )
            rebuilt_pipeline = NamedTransformer(
                ColumnTransformer(
                    transformers,
                    remainder='drop'
                )
            )
            return rebuilt_pipeline
        else:
            return preproc_pipeline
        

class _DtypeBasedPreprocessingSklearn:
    """build preprocessing pipeline based on data schema
    """
    GROUP_COL = ENTITY_CFG.entity_key
    
    def __call__(self, hyper_mode: HyperMode) -> TransformerMixin:
        """build preprocessing pipeline

        :param HyperMode hyper_mode: hyper tuning mode and params
        :return TransformerMixin: sklearn-compatible transformer
        """
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
                trans = Pipeline([
                    # binarize
                    ('binarize', BinaryConverter(
                        pos_values = [1, True, 'Y'], 
                        keep_na = True
                    )),
                    # imputation
                    ('impute', MyImputer(
                        strategy = 'constant', 
                        fill_value = -1, 
                        add_indicator = True
                    ))
                ])
            elif ibis_dtype.is_numeric():
                trans = Pipeline([
                    # use 1%-99% quantile capping
                    ('clip', OutlierClipper(
                        strategy = 'quantile', 
                        quantile_range = (1, 99)
                    )),
                    # median value imputation
                    ('impute', MyImputer(
                        strategy = 'median', 
                        add_indicator = False
                    )),
                    # normalization
                    ('normalize', StandardScaler())
                ])
            elif ibis_dtype.is_string() and not ibis_dtype.is_date() and not ibis_dtype.is_timestamp():
                trans = Pipeline([
                    # convert to string type
                    #('int_stringfy', FunctionTransformer(func=intStringfy)),
                    # impute with Missing
                    ('impute', MyImputer(
                        strategy = 'constant', 
                        fill_value = 'Missing', 
                        add_indicator = False
                    )),
                    # reduce categories if too much
                    ('bucket', BucketCategValue(threshold = 'auto')),
                    # one hot encoding with as most 10 categories
                    ('ohe', OneHotEncoder(
                        sparse = False, 
                        handle_unknown = 'ignore'
                    ))  # and one hot encode the categories
                ])
            else:
                continue
                # TODO: need to add them to the list -- find a universal way of preprocessing unknown type columns

            transformers.append(
                (col, trans, make_present_col_selector([col] ))
            )
  
        pipe = NamedTransformer(
            ColumnTransformer(
                transformers = transformers,
                remainder = 'drop',
                #n_jobs = -1
            )
        )
        return pipe
            


@dataclass
class PreprocessParam:
    mode: Literal['Free', 'EDA'] = 'EDA'
    template_id: str | None = None
    eda_model_id: str | None = None

class PreprocessingBaseSklearn:
    FREE_MODE_TEMP_IDS = {
        'DtypeBased' : _DtypeBasedPreprocessingSklearn
    }
    
    def __init__(self, preprocessing_pipe: TransformerMixin):
        self.setPipe(preprocessing_pipe)

    def setPipe(self, preprocessing_pipe: TransformerMixin):
        self.__prep_pipe = preprocessing_pipe

    def getPipe(self) -> TransformerMixin:
        return self.__prep_pipe

    @classmethod
    def build(cls, hyper_mode: HyperMode, param: PreprocessParam) -> TransformerMixin:
        """build preprocess pipeline (static computation graph)

        :param HyperMode hyper_mode: hyper tuning mode and params
        :param PreprocessParam param: other configurations
        :return TransformerMixin: sklearn transformer
        """
        if param.mode == 'EDA':
            if param.eda_model_id is None:
                raise ValueError("Must provide eda_model_id if use EDA mode preprocessing")
            
            preproc = _EdaBasedPreprocessingSklearn(
                eda_model_id = param.eda_model_id
            )
        else:
            # free mode
            if param.template_id is None:
                raise ValueError("Must provide template_id if use Free mode preprocessing")
            if param.template_id not in cls.FREE_MODE_TEMP_IDS:
                raise ValueError(f"{param.template_id} not found, please registry first to code")
            
            preproc = cls.FREE_MODE_TEMP_IDS.get(
                param.template_id
            )() # free mode will not have ability to pass parameter
            
        pipe = preproc(hyper_mode)
        return pipe
