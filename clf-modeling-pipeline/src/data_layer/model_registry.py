from datetime import datetime, timezone
from typing import List, Literal, Tuple
import pymongo
import json
from luntaiDs.ProviderTools.mongo.serving import _BaseModelRegistryMongo
from luntaiDs.ModelingTools.FeatureEngineer.preprocessing import TabularPreprocModel

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