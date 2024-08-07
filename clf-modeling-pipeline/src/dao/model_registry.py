from datetime import datetime, timezone
from typing import List, Literal, Tuple
from pathlib import Path
import ibis
import pymongo
import json
import mlflow
import optuna
import shap
import tempfile
import joblib
from PIL import Image
from luntaiDs.ProviderTools.mongo.serving import _BaseModelRegistryMongo
from luntaiDs.ProviderTools.mlflow.dtyper import ibis_schema_2_mlflow_schema
from luntaiDs.ModelingTools.FeatureEngineer.preprocessing import TabularPreprocModel
from src.model_layer.composite import CompositePipeline, HyperMode

class EdaPreprocRegistryMongo(_BaseModelRegistryMongo):
    """EDA based preprocessing model registry
    not that:
    1. model metadata (registry) is based on MongoDB (db: self.db, collection: self.collection)
    2. model data (profiling result) can be saved to anywhere defined in this class,
        while this class choose to also save model data to MongoDB as well (in separate collection as 1.)
        - (db: self.db, collection: 'data_eda_preproc')
    3. this implementation will directly accept TabularPreprocModel object, if wish to accept serialized JSON, nee
        a separate implementation (e.g., in micro-service scenario, will send/receive JSON object instead)
    """
    DATA_COLLECTION = 'data_eda_preproc'
    
    def delete_model_files(self, model_id: str):
        """delete model related files/data

        :param str model_id: model id to be deleted
        """
        collection = self._mongo[self.db][self.DATA_COLLECTION]
        collection.delete_many(
            {
                'model_id': model_id
            }, # find matching record, if any
        )

    def load_model_by_config(self, config: dict) -> TabularPreprocModel:
        """load the model using configuration file

        :param dict config: configuration for given model to help load model data back
        :return TabularPreprocModel: the tabular preprocessing model, a wrapper around EDA result + preproc params
        """
        model_id = config['model_id']
        db = config.get('db', self.db)
        collect = config.get('collection', self.DATA_COLLECTION)
        collection = self._mongo[db][collect]
        records = (
            collection
            .find(
                {'model_id': model_id}, # matching condition
                {'_id': 0}, # drop id column
                sort = [( '_id', pymongo.DESCENDING )] # in case multiple, find the latest record
            )
        )
        deserialized = []
        for record in records:
            preproc = record.get('preproc')
            stat_obj = {
                'colname' : record['colname'],
                'attr': record['attr'],
                'summary': record['summary'],
            }
            deserialized.append(dict(
                constructor = record['model_construct'],
                stat_obj = stat_obj,
                preproc = preproc
            ))

        return TabularPreprocModel.deserialize(deserialized)

    def save_model_and_generate_config(self, model_id: str, tpm: TabularPreprocModel, 
            data_source: Literal['data_registry', 'ad_hoc'] = 'ad_hoc', data_config: dict | None = None) -> dict:
        """save model and generate configuration for this model

        :param str model_id: model id to be generated 
            (can be linked to a specific data id of training data)
        :param TabularPreprocModel tpm: the tabular preprocessing model
        :param str data_source: where the data come from
        :param dict data_config: if given, will link to specific data id
        :return dict: generate the config for model metadata
        """
        collection = self._mongo[self.db][self.DATA_COLLECTION]
        
        # serialize tabular stat result
        serialized = tpm.serialize()
        records = []
        for model_preproc_js in serialized:
            stat_obj = model_preproc_js['stat_obj']
            preproc = model_preproc_js['preproc']
            content = {
                'model_id' : model_id,
                'colname' : stat_obj['colname'],
                'stat_construct': stat_obj['constructor'],
                'model_construct': model_preproc_js['constructor'],
                'attr': stat_obj['attr'],
                'summary': stat_obj['summary'],
                'preproc': preproc
            }
            records.append(
                # need to load and dump to make sure it is mongodb compatible
                json.loads(json.dumps(content))
            )

        collection.insert_many(records)
        
        return {
            'model_id' : model_id,
            'data_source': data_source,
            'data_config' : data_config,
            'db' : self.db,
            'collection' : self.DATA_COLLECTION,
            'last_update_ts' : datetime.now(timezone.utc),
        }
        
class EdaFeatureSelRegistryMongo(_BaseModelRegistryMongo):
    """EDA backed feature selection model registry
    not that:
    1. model metadata (registry) is based on MongoDB (db: self.db, collection: self.collection)
    2. model data (detailed result) can be saved to anywhere defined in this class,
        while this class choose to also save model data to MongoDB as well (in separate collection as 1.)
        - (db: self.db, collection: 'data_eda_fsel')
    """
    DATA_COLLECTION = 'data_eda_fsel'
    
    def delete_model_files(self, model_id: str):
        """delete model related files/data

        :param str model_id: model id to be deleted
        """
        collection = self._mongo[self.db][self.DATA_COLLECTION]
        collection.delete_many(
            {
                'model_id': model_id
            }, # find matching record, if any
        )
        
    def load_model_by_config(self, config: dict) -> Tuple[List[str], str]:
        """load the model using configuration file

        :param dict config: configuration for given model to help load model data back
        :return Tuple[List[str], str]: (list of selected features, description)
        """
        model_id = config['model_id']
        db = config.get('db', self.db)
        collect = config.get('collection', self.DATA_COLLECTION)
        collection = self._mongo[db][collect]
        record = (
            collection
            .find_one(
                {'model_id': model_id}, # matching condition
                {'_id': 0}, # drop id column
                sort = [( '_id', pymongo.DESCENDING )] # in case multiple, find the latest record
            )
        )
        return record['selected'], record['description']
    
    def save_model_and_generate_config(self, model_id: str, selected_cols: List[str], description: str,
            data_source: Literal['data_registry', 'ad_hoc'] = 'ad_hoc', data_config: dict | None = None) -> dict:
        """save model and generate configuration for this model

        :param str model_id: model id to be generated 
            (can be linked to a specific data id of training data)
        :param List[str] selected_cols: selected features
        :param str description: a description on how features were selected
        :param str data_source: where the data come from
        :param dict data_config: if given, will link to specific data id
        :return dict: generate the config for model metadata
        """
        content = {
            'model_id' : model_id,
            'selected': selected_cols,
            'description': description
        }
        collection = self._mongo[self.db][self.DATA_COLLECTION]
        collection.replace_one(
            filter = {
                'model_id': 'model_id',
            }, # find matching record, if any
            replacement = content,
            upsert = True # update if found, insert if not found
        )
        
        return {
            'model_id' : model_id,
            'data_source': data_source,
            'data_config' : data_config,
            'db' : self.db,
            'collection' : self.DATA_COLLECTION,
            'last_update_ts' : datetime.now(timezone.utc),
        }


class MlflowCompositePipeline(mlflow.pyfunc.PythonModel, CompositePipeline):
    def predict(self, context, model_input, params):
        return self.inference(model_input)
    
class MlflowMongoWholeModelRegistry(_BaseModelRegistryMongo):
    def delete_model_files(self, model_id: str):
        """delete model related files/data
            note for mlflow implementation, only registry record will be deleted,
            but not the underlying logged files

        :param str model_id: model id to be deleted
        """
        client = mlflow.MlflowClient()
        client.delete_registered_model(name = model_id)

    def load_model_by_config(self, config: dict) -> MlflowCompositePipeline:
        """load the model using configuration file

        :param dict config: configuration for given model to help load model data back
        :return MlflowCompositePipeline: the mlflow implementation of composite pipeline
        """
        run_id = config['tracking_info']['model_info']['model_uri']
        loaded_model = mlflow.pyfunc.load_model(run_id)
        return loaded_model.unwrap_python_model()
    
    def load_mlflow_pyfunc_model(self, model_id: str) -> mlflow.pyfunc.PythonModel:
        """load the mlflow wrapped python flavor model

        :param str model_id: model id to be loaded
        :return mlflow.pyfunc.PythonModel: the mlflow wrapped flavor model
        """
        config = self.get_model_config(model_id = model_id)
        run_id = config['tracking_info']['model_info']['model_uri']
        loaded_model = mlflow.pyfunc.load_model(run_id)
        return loaded_model
    
    def load_shap_explainer(self, model_id: str) -> shap.Explainer:
        """load back shap explainer saved during training/registry

        :param str model_id: model id for the relevant shap explainer
        :return shap.Explainer: the shap explainer trained on the training data
        """
        from mlflow import MlflowClient
        
        config = self.get_model_config(model_id = model_id)
        run_id = config['tracking_info']['best_run_id']
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            client = MlflowClient()
            client.download_artifacts(
                run_id = run_id,
                path = 'shaps/explainer.sav',
                dst_path = tmp_dir
            )
            with open(Path(tmp_dir) / 'shaps/explainer.sav', "rb") as f:
                explainer = joblib.load(f)
                
        return explainer
    
    def load_shap_global_plots(self, model_id: str) -> Tuple[Image.Image, Image.Image]:
        """load shap global bar and beeswarm chart on the training set

        :param str model_id: model id for the relevant shap explainer
        :return Tuple[Image.Image, Image.Image]: (bar plot, beeswarm plot)
        """
        from mlflow import MlflowClient
        
        config = self.get_model_config(model_id = model_id)
        run_id = config['tracking_info']['best_run_id']
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            client = MlflowClient()
            client.download_artifacts(
                run_id = run_id,
                path = 'shaps/global_bar.png',
                dst_path = tmp_dir
            )
            client.download_artifacts(
                run_id = run_id,
                path = 'shaps/global_beeswarm.png',
                dst_path = tmp_dir
            )
            bar_img = Image.open(Path(tmp_dir) / 'shaps/global_bar.png')
            beeswar_img = Image.open(Path(tmp_dir) / 'shaps/global_beeswarm.png')
            
        return bar_img, beeswar_img
            

    def save_model_and_generate_config(self, model_id: str, data_id: str,
            X_train: ibis.expr.types.Table,
            cp: MlflowCompositePipeline, hyper_mode: HyperMode, 
            ) -> dict:
        """save model and generate configuration for this model

        :param str model_id: model id to be generated 
            (will be used as mlflow model deployment name)
        :param str data_id: data id which used to train the model
        :param ibis.expr.types.Table X_train: training data in ibis dataframe format
        :param HyperMode hyper_mode: hyper parameter tuning info
        :param MlflowCompositePipeline cp: the composite pipeline object (mlflow implementation)
        """
        # retrieve optuna study
        study_name = hyper_mode.attrs['study_name']
        study = optuna.study.load_study(
            study_name=study_name, 
            storage=hyper_mode.hyper_storage
        )
        best_run_id = study.user_attrs.get('mlflow_best_run_id')
        
        # log model to tracking server
        with mlflow.start_run(run_id = best_run_id) as run:
            X_train_pd = X_train.to_pandas()
            model_info = mlflow.pyfunc.log_model(
                python_model=cp, 
                artifact_path="composite_pipeline", 
                signature = mlflow.models.ModelSignature(
                    inputs = ibis_schema_2_mlflow_schema(X_train.schema()),
                    outputs = mlflow.models.infer_signature(
                        model_output = cp.inference(X_train_pd.sample(1))
                    ).outputs
                ),
                metadata=cp.getLoggingParams()
            )
            # log pipeline parameters and attributes
            mlflow.log_dict(
                dictionary=cp.getLoggingParams(),
                artifact_file='extras/pipeline_params.json'
            )
            mlflow.log_dict(
                dictionary=cp.getLoggingAttrs(),
                artifact_file='extras/pipeline_attrs.json'
            )
            # log optuna tuning table
            mlflow.log_table(
                data=study.trials_dataframe(),
                artifact_file='extras/tuning_result.json'
            )
            # log model structure
            mlflow.log_text(
                text=cp.renderStructureHTML(),
                artifact_file='extras/structure.html'
            )
            # log shap explainer
            with tempfile.TemporaryDirectory() as tmp_dir:
                X_train_premodel = cp.transformPreModel(X_train_pd)
                explainer = cp.getShapExplainer(
                    X_train_premodel, 
                    premodel = True
                )
            
                # save to local temp file 1st
                tmp_path = Path(tmp_dir) / "explainer.sav"
                with open(tmp_path, "wb") as f:
                    joblib.dump(explainer, f)
                
                mlflow.log_artifact(
                    local_path=tmp_path.as_posix(),
                    artifact_path='shaps'
                )
                
                # save shap global plots
                from matplotlib import pyplot as plt
                from io import BytesIO
                from PIL import Image
                
                exps = explainer(X_train_premodel)
                
                f, ax = plt.subplots(figsize = (25, 10), dpi = 72)
                bar_plot = shap.plots.bar(
                    exps, 
                    max_display = 50, 
                    show = False, 
                    ax = ax
                )
                mlflow.log_figure(
                    figure = f,
                    artifact_file='shaps/global_bar.png'
                )
                
                f, ax = plt.subplots(figsize = (25, 10), dpi = 72)
                beeswarm_plot = shap.plots.beeswarm(
                    exps, 
                    max_display = 50, 
                    show = False,
                    plot_size = (25, 10)
                )
                mlflow.log_figure(
                    figure = f,
                    artifact_file='shaps/global_beeswarm.png'
                )
        
        # register model to mlflow registry
        model_version = mlflow.register_model(
            model_uri = model_info.model_uri,
            name = model_id,
        )
            
        run_info: mlflow.entities.RunInfo = mlflow.get_run(run_id=best_run_id).info
        return {
            'model_id': model_id,
            'data_id': data_id,
            'params': cp.getLoggingParams(),
            # optuna info
            'tuning_info': {
                'study_id': hyper_mode.hyper_storage.get_study_id_from_name(
                    study_name=study_name
                ),
                'study_name': study_name,
                'direction': study.direction.value, # int enum
                'best_trial': {
                    'number': study.best_trial.number,
                    'value': study.best_trial.value,
                    'start': study.best_trial.datetime_start,
                    'end': study.best_trial.datetime_complete,
                }
            },
            # mlflow info
            'tracking_info': {
                'best_run_id': best_run_id,
                'best_run_info': {
                    'experiment_id': run_info.experiment_id,
                    'run_id': run_info.run_id,
                    'run_name': run_info.run_name,
                    'user_id': run_info.user_id,
                    'artifact_uri': run_info.artifact_uri,
                    
                },
                'model_info' : {
                    'model_uri': model_info.model_uri,
                    'created_time': model_info.utc_time_created,
                    'artifact_path': model_info.artifact_path,
                    'metadata': model_info.metadata,
                    'flavors': model_info.flavors,
                    'signature': model_info.signature_dict
                },
                'registry_info' : {
                    'name' : model_version.name,
                    'aliases' : model_version.aliases,
                    'creation_timestamp': model_version.creation_timestamp,
                    'last_updated_timestamp': model_version.last_updated_timestamp,
                    'current_stage': model_version.current_stage,
                    'run_id' : model_version.run_id,
                    'run_link': model_version.run_link,
                    'source': model_version.source,
                    'version': model_version.version
                }
            }
            
        }