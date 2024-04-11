import os
import sys
from typing import List
sys.path.append(os.getcwd())

import streamlit as st
from luntaiDs.ModelingTools.Explore.summary import BinaryStatAttr, NominalCategStatAttr, \
        OrdinalCategStatAttr, NumericStatAttr, BinaryStatObj, NumericStatObj, \
            NominalCategStatObj, OrdinalCategStatObj, BaseStatAttr
from luntaiDs.ModelingTools.FeatureEngineer.preprocessing import BaseFeaturePreproc, BinaryFeaturePreproc, \
    NumericFeaturePreproc, NominalCategFeaturePreproc, OrdinalCategFeaturePreproc
    
from src.services.eda.apis.apis import _BaseDataManager

def get_existing_data_ids_from_data_registry() -> List[str]:
    return _BaseDataManager.get_existing_data_ids_from_data_registry()

def get_existing_eda_model_ids() -> List[str]:
    return _BaseDataManager.get_existing_eda_model_ids()

def get_existing_eda_data_source_config(eda_model_id: str) -> dict:
    return _BaseDataManager.get_existing_eda_data_source_config(eda_model_id)

def get_existing_eda_model_profile(eda_model_id: str) -> list:
    return _BaseDataManager.get_existing_eda_model_profile(eda_model_id)

def get_existing_buckets_from_obj_storage() -> List[str]:
    return _BaseDataManager.get_existing_buckets_from_obj_storage()

def add_data_from_data_registry(data_id: str, is_train: bool = True):
    st.session_state['data_config'] = {
        'source' : 'data_registry',
        'config': {
            'data_id' : data_id,
            'is_train' : is_train
        }
    }
    
def add_data_from_query(query: str):
    st.session_state['data_config'] = {
        'source' : 'ad_hoc_query',
        'config': {
            'query' : query,
        }
    }
    
def add_data_from_obj_storage(bucket: str, file_path: str):
    st.session_state['data_config'] = {
        'source' : 'ad_hoc_file',
        'config': {
            'bucket' : bucket,
            'file_path' : file_path
        }
    }
    
    
    
def save_or_clear_selected_eda_model_id(eda_model_id: str | None = None):
    if eda_model_id:
        st.session_state['selected_eda_model_id'] = eda_model_id
    else:
        st.session_state['selected_eda_model_id'] = None

def reset_profile():
    st.session_state['standalone_config'] = {}
    
def update_profile(tpm_serialized: list):
    reset_profile()
    
    for preproc_model_js in tpm_serialized:
        col = preproc_model_js['colname']
        constructor = preproc_model_js['constructor']
        if constructor == 'BinaryFeaturePreprocModel':
            dtype = 'Binary'
            attr_cls = BinaryStatAttr
            preproc_cls = BinaryFeaturePreproc
        if constructor == 'OrdinalFeaturePreprocModel':
            dtype = 'Ordinal'
            attr_cls = OrdinalCategStatAttr
            preproc_cls = OrdinalCategFeaturePreproc
        if constructor == 'NominalFeaturePreprocModel':
            dtype = 'Nominal'
            attr_cls = NominalCategStatAttr
            preproc_cls = NominalCategFeaturePreproc
        if constructor == 'NumericFeaturePreprocModel':
            dtype = 'Numeric'
            attr_cls = NumericStatAttr
            preproc_cls = NumericFeaturePreproc
        
        stat_attr = preproc_model_js['stat_obj']['attr']
        preproc = preproc_model_js['preproc']

        st.session_state['standalone_config'][col] = {
            'dtype' : dtype,
            'stat_attr': attr_cls.deserialize(stat_attr),
            'preproc': preproc_cls.deserialize(preproc)
        }

def delete_eda_model(eda_model_id: str):
    _BaseDataManager.delete_eda_model(eda_model_id)

    
model_source_ = st.radio(
    label = 'What you are trying to do',
    options = ['Start New EDA Modeling', 'Fetch Existing EDA Model'],
    index = 0,
    horizontal = True
)
st.divider()

if model_source_ == 'Start New EDA Modeling':
    reset_profile()
    save_or_clear_selected_eda_model_id()

    data_source_ = st.radio(
        label = 'Where your data come from?',
        options = ['Data Registry', 'Query to Data Warehouse', 'File From Obj Storage'],
        index = 0,
        horizontal = True
    )

    if data_source_ == 'Data Registry':
        data_id_ = st.selectbox(
            label = 'Select Data Id',
            options = get_existing_data_ids_from_data_registry(),
        )
        is_train_ = st.toggle(
            label = 'From Train Set?',
            value = True
        )
        add_data_from_data_registry(
            data_id = data_id_,
            is_train = is_train_
        )

    elif data_source_ == 'Query to Data Warehouse':
        query_ = st.text_area(
            label = 'SQL Query',
            placeholder = 'SELECT * FROM ...'
        )
        if 'data_config' in st.session_state:
            st.session_state.pop('data_config')
        if query_:
            add_data_from_query(
                query = query_
            )
            
    elif data_source_ == 'File From Obj Storage':
        bucket_ = st.selectbox(
            label = 'Which Bucket',
            options = get_existing_buckets_from_obj_storage()
        )
        file_path_ = st.text_input(
            label = "File Path (.parquet/csv)",
            placeholder = '/path/to/file.parquet'
        )
        if 'data_config' in st.session_state:
            st.session_state.pop('data_config')
        if file_path_:
            add_data_from_obj_storage(
                bucket = bucket_,
                file_path = file_path_
            )
    
    if 'data_config' in st.session_state:       
        st.json(st.session_state['data_config'])

else:
    eda_model_id_ = st.selectbox(
        label = 'Select EDA Model Id',
        options = get_existing_eda_model_ids(),
    )
    save_or_clear_selected_eda_model_id(eda_model_id_)
    
    data_source_config_ = get_existing_eda_data_source_config(eda_model_id_)
    tpm_serialized = get_existing_eda_model_profile(eda_model_id_)
    update_profile(tpm_serialized)
    
    st.json(data_source_config_)

    data_source_ = data_source_config_['source']
    if data_source_ == 'data_registry':
        add_data_from_data_registry(
            data_id = data_source_config_['config']['data_id'],
            is_train = data_source_config_['config']['is_train']
        )
    if data_source_ == 'ad_hoc_query':
        add_data_from_query(
            query = data_source_config_['config']['query'],
        )
    if data_source_ == 'ad_hoc_file':
        add_data_from_obj_storage(
            bucket = data_source_config_['config']['bucket'],
            file_path = data_source_config_['config']['file_path']
        )
        
    st.button(
        label = 'Delete EDA Model',
        type = 'primary',
        on_click = delete_eda_model,
        kwargs = dict(
            eda_model_id = eda_model_id_,
        )
    )
        
        