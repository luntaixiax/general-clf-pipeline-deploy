import json
import os
import sys
from typing import List, Literal
sys.path.append(os.getcwd())
import streamlit as st

# TODO: allow luntaiDs dataclass (data container) but do not allow this project's object
from luntaiDs.ModelingTools.Explore.summary import BinaryStatAttr, NominalCategStatAttr, \
        OrdinalCategStatAttr, NumericStatAttr, BinaryStatObj, NumericStatObj, \
            NominalCategStatObj, OrdinalCategStatObj, BaseStatAttr
from luntaiDs.ModelingTools.FeatureEngineer.preprocessing import BaseFeaturePreproc, BinaryFeaturePreproc, \
    NumericFeaturePreproc, NominalCategFeaturePreproc, OrdinalCategFeaturePreproc, TabularPreprocModel
from src.services.eda.apis.plots import get_nominal_ordinal_plots, get_binary_plots, get_numeric_plots, \
        plot_table_profiling_to_html_string
# TODO: will replace the below imports into micro service
from src.services.eda.apis.apis import RemoteDataManagerForRegistry, RemoteDataManagerForQuery


def get_data_manager():
    data_config = st.session_state['data_config']
    if data_config['source'] == 'data_registry':
        return RemoteDataManagerForRegistry(
            data_id = data_config['config']['data_id'],
            is_train = data_config['config']['is_train']
        )
    elif data_config['source'] == 'ad_hoc_query':
        return RemoteDataManagerForQuery(
            query = data_config['config']['query']
        )


@st.cache_data
def get_or_train_numeric_fsummary(data_config: dict, 
        col: str, attr: NumericStatAttr) -> NumericStatObj:
    return get_data_manager().train_numeric_stat(
        column = col,
        attr = attr
    )
    
@st.cache_data
def get_or_train_nominal_fsummary(data_config: dict, 
        col: str, attr: NominalCategStatAttr) -> NominalCategStatObj:
    return get_data_manager().train_nominal_stat(
        column = col,
        attr = attr
    )
    
@st.cache_data
def get_or_train_binary_fsummary(data_config: dict, 
        col: str, attr: BinaryStatAttr) -> BinaryStatObj:
    return get_data_manager().train_binary_stat(
        column = col,
        attr = attr
    )
    
@st.cache_data
def get_or_train_ordinal_fsummary(data_config: dict, 
        col: str, attr: OrdinalCategStatAttr) -> OrdinalCategStatObj:
    return get_data_manager().train_nominal_stat(
        column = col,
        attr = attr
    )

def save_update_column_profile(col: str, dtype: Literal['Binary', 'Nominal', 'Ordinal', 'Numeric'], 
        stat_attr: BaseStatAttr, preproc: BaseFeaturePreproc):
    # always update the profile
    st.session_state['standalone_config'][col] = {
        'dtype' : dtype,
        'stat_attr': stat_attr,
        'preproc': preproc
    }
    
def serialize_profile() -> dict:
    s = {}
    for col, config in st.session_state['standalone_config'].items():
        s[col] = {
            'dtype' : config['dtype'],
            'attr' : config['stat_attr'].serialize(),
            'preproc' : config['preproc'].serialize(),
        }
    return s
    

def train_and_save_eda_model(eda_model_id: str):
    tpm_list = get_data_manager().train_and_save_eda_model(
        eda_model_id = eda_model_id,
        serialized_profile = serialize_profile()
    )
    st.session_state['tpm_list_downloadable'] = tpm_list
    
def remove_profile(col: str):
    if col in st.session_state['standalone_config']:
        del st.session_state['standalone_config'][col]
        
def plot_html():
    if 'tpm_list_downloadable' in st.session_state:
        tpm_list = st.session_state['tpm_list_downloadable']
        tpm = TabularPreprocModel.deserialize(tpm_list)
        plot = plot_table_profiling_to_html_string(tpm)
        return plot
        
if 'data_config' in st.session_state:
    if 'standalone_config' not in st.session_state:
        # initialize
        st.session_state['standalone_config'] = {}
            
    with st.popover("Sample Data"):
        st.dataframe(get_data_manager().get_data_preview())
    
    with st.sidebar:
        selected_feature_ = st.selectbox(
            label = 'Select Feature',
            options = get_data_manager().get_columns()
        )
        # load from session state first
        saved_col_config = st.session_state['standalone_config'].get(selected_feature_)
        if saved_col_config is not None:
            guessed_dtype_ = saved_col_config['dtype']
        else:
            guessed_dtype_ = get_data_manager().guess_col_dtype(selected_feature_)
            st.info(f"Feature Never Seen, guessed Dtype: {guessed_dtype_}")
    
    # options column
    feature_stat_option_cols = st.columns(2)
    with feature_stat_option_cols[0]:
        type_options = ['Binary', 'Nominal', 'Ordinal', 'Numeric']
        dtype_choice_ = st.selectbox(
            label = 'Feature Dtype',
            options = type_options,
            index = type_options.index(guessed_dtype_),
        )
    # stat attributes
    
    if dtype_choice_ == 'Binary':
        # important! must be same type to load back params, or will load to wrong class
        if guessed_dtype_ == dtype_choice_ and saved_col_config is not None:
            # load from cache
            loaded_attr = saved_col_config['stat_attr']
            int_dtype_v = loaded_attr.int_dtype_
            na_to_pos_v = loaded_attr.na_to_pos_
            pos_values_v = loaded_attr.pos_values_
        else:
            # initial value
            int_dtype_v = False
            na_to_pos_v = False
            pos_values_v = None
        
        with feature_stat_option_cols[1]:
            pos_values_select_ = st.multiselect(
                label = "Select values for class 1",
                options = get_data_manager().get_value_list(column = selected_feature_),
                default = pos_values_v
            )
            
            cols_binary_ = st.columns([2, 3])
            with cols_binary_[0]:
                int_dtype_check_ = st.toggle(
                    label = 'Int Dtype',
                    value = int_dtype_v,
                )
            with cols_binary_[1]:
                na_to_pos_check_ = st.toggle(
                    label = 'Missing as class 1',
                    value = na_to_pos_v,
                )
    
        
        # preprocess options
        attr = BinaryStatAttr(
            int_dtype_ = int_dtype_check_,
            pos_values_ = pos_values_select_,
            na_to_pos_ = na_to_pos_check_
        )
        preproc = BinaryFeaturePreproc()
        
        btn_cols = st.columns(2)
        with btn_cols[0]:
            st.button(
                label = 'Save/Update',
                type = 'primary',
                on_click = save_update_column_profile,
                kwargs = dict(
                    col = selected_feature_,
                    dtype = dtype_choice_,
                    stat_attr = attr,
                    preproc = preproc
                )
            )
        with btn_cols[1]:
            st.button(
                label = 'Remove',
                type = 'secondary',
                on_click = remove_profile,
                kwargs = dict(
                    col = selected_feature_,
                )
            )
        
        # display and info section (no data saving)
        summary_ = get_or_train_binary_fsummary(
            data_config = st.session_state['data_config'],
            col = selected_feature_,
            attr = attr
        )
        
        # plot
        categ_tabs = st.tabs(['Composite Donut', 'Category Distribution'])
        plots = get_binary_plots(summary_)
        with categ_tabs[0]:
            st.bokeh_chart(
                figure = plots['donut'], 
                use_container_width = False
            )
        with categ_tabs[1]:
            st.bokeh_chart(
                figure = plots['distr'], 
                use_container_width = False
            )
        
            
    elif dtype_choice_ == 'Nominal':
        # important! must be same type to load back params, or will load to wrong class
        if guessed_dtype_ == dtype_choice_ and saved_col_config is not None:
            # load from cache
            loaded_attr = saved_col_config['stat_attr']
            int_dtype_v = loaded_attr.int_dtype_
            max_categories_v = loaded_attr.max_categories_
        else:
            # initial value
            int_dtype_v = False
            max_categories_v = 10
        
        with feature_stat_option_cols[1]:
            max_categories_inp_ = st.number_input(
                label = 'Max # of categories',
                min_value = 3,
                max_value = 500,
                value = max_categories_v,
                step = 1
            )
            int_dtype_check_ = st.checkbox(
                label = 'Int Dtype',
                value = int_dtype_v,
            )
        
        # preprocess widges
        # important! must be same type to load back params, or will load to wrong class
        if guessed_dtype_ == dtype_choice_ and saved_col_config is not None:
            # load from cache
            loaded_preproc = saved_col_config['preproc']
            impute_value_v = loaded_preproc.impute_value
            bucket_strategy_v = loaded_preproc.bucket_strategy
            encode_strategy_v = loaded_preproc.encode_strategy
        else:
            impute_value_v = 'Other'
            bucket_strategy_v = None
            encode_strategy_v = 'ohe'
            
        btn_cols = st.columns(3)
        with btn_cols[0]:
            with st.popover("Preprocess Instruction"):
                if impute_value_v in [None, 'most_frequent']:
                    impute_method_v = impute_value_v
                else:
                    impute_method_v = 'custom'
                
                impute_method_ = st.radio(
                    label = "Imputation Method",
                    options = [None, 'most_frequent', 'custom'],
                    index = [None, 'most_frequent', 'custom'].index(impute_method_v) # default to Custom
                )
                if impute_method_ == 'custom':
                    impute_value_ = st.text_input(
                        label = 'Impute Missing Value as',
                        value = impute_value_v
                    )
                else:
                    impute_value_ = impute_method_
                
                bucket_strategy_ = st.selectbox(
                    label = 'Bucketize Strategy',
                    options = [None, 'freq', 'correlation'],
                    index = [None, 'freq', 'correlation'].index(bucket_strategy_v)
                )
                encode_strategy_ = st.selectbox(
                    label = 'Encoding Strategy',
                    options = ['ohe', 'ce', 'woe'],
                    index = ['ohe', 'ce', 'woe'].index(encode_strategy_v)
                )
        
        # preprocess options
        attr = NominalCategStatAttr(
            int_dtype_ = int_dtype_check_,
            max_categories_ = max_categories_inp_
        )
        preproc = NominalCategFeaturePreproc(
            impute_value = impute_value_,
            bucket_strategy = bucket_strategy_,
            encode_strategy = encode_strategy_,
        )
        
            
        with btn_cols[1]:
            st.button(
                label = 'Save/Update',
                type = 'primary',
                on_click = save_update_column_profile,
                kwargs = dict(
                    col = selected_feature_,
                    dtype = dtype_choice_,
                    stat_attr = attr,
                    preproc = preproc
                )
            )
        with btn_cols[2]:
            st.button(
                label = 'Remove',
                type = 'secondary',
                on_click = remove_profile,
                kwargs = dict(
                    col = selected_feature_,
                )
            )
        
        # display and info section (no data saving)
        summary_ = get_or_train_nominal_fsummary(
            data_config = st.session_state['data_config'],
            col = selected_feature_,
            attr = attr
        )
        
        # plot
        categ_tabs = st.tabs(['Composite Donut', 'Category Distribution'])
        plots = get_nominal_ordinal_plots(summary_)
        with categ_tabs[0]:
            st.bokeh_chart(
                figure = plots['donut'], 
                use_container_width = False
            )
        with categ_tabs[1]:
            st.bokeh_chart(
                figure = plots['distr'], 
                use_container_width = False
            )
        
    elif dtype_choice_ == 'Ordinal':
        # important! must be same type to load back params, or will load to wrong class
        if guessed_dtype_ == dtype_choice_ and saved_col_config is not None:
            # load from cache
            loaded_attr = saved_col_config['stat_attr']
            int_dtype_v = loaded_attr.int_dtype_
            categories_v = loaded_attr.categories_
        else:
            # initial value
            int_dtype_v = False
            categories_v = None
        
        with feature_stat_option_cols[1]:
            category_values_select_ = st.multiselect(
                label = "Rank Values for Ordinal Var",
                options = get_data_manager().get_value_list(column = selected_feature_),
                default = categories_v
            )
            int_dtype_check_ = st.checkbox(
                label = 'Int Dtype',
                value = int_dtype_v,
            )
        
        # preprocess widges
        # important! must be same type to load back params, or will load to wrong class
        if guessed_dtype_ == dtype_choice_ and saved_col_config is not None:
            # load from cache
            loaded_preproc = saved_col_config['preproc']
            impute_v = loaded_preproc.impute
            standardize_v = loaded_preproc.standardize
        else:
            impute_v = True
            standardize_v = False
            
        btn_cols = st.columns(3)
        with btn_cols[0]:
            with st.popover("Preprocess Instruction"):
                impute_toggle_ = st.toggle(
                    label = 'Apply Imputation',
                    value = impute_v,
                )
                standard_toggle_ = st.toggle(
                    label = 'Apply Standardization',
                    value = standardize_v,
                )
            
        # preprocess options
        attr = OrdinalCategStatAttr(
            int_dtype_ = int_dtype_check_,
            categories_ = category_values_select_
        )
        preproc = OrdinalCategFeaturePreproc(
            impute = impute_toggle_,
            standardize = standard_toggle_
        )
        
        with btn_cols[1]:
            st.button(
                label = 'Save/Update',
                type = 'primary',
                on_click = save_update_column_profile,
                kwargs = dict(
                    col = selected_feature_,
                    dtype = dtype_choice_,
                    stat_attr = attr,
                    preproc = preproc
                )
            )
        with btn_cols[2]:
            st.button(
                label = 'Remove',
                type = 'secondary',
                on_click = remove_profile,
                kwargs = dict(
                    col = selected_feature_,
                )
            )
        
        # display and info section (no data saving)
        summary_ = get_or_train_ordinal_fsummary(
            data_config = st.session_state['data_config'],
            col = selected_feature_,
            attr = attr
        )
        
        # plot
        categ_tabs = st.tabs(['Composite Donut', 'Category Distribution'])
        plots = get_nominal_ordinal_plots(summary_)
        with categ_tabs[0]:
            st.bokeh_chart(
                figure = plots['donut'], 
                use_container_width = False
            )
        with categ_tabs[1]:
            st.bokeh_chart(
                figure = plots['distr'], 
                use_container_width = False
            )
        
    elif dtype_choice_ == 'Numeric':
        # important! must be same type to load back params, or will load to wrong class
        if guessed_dtype_ == dtype_choice_ and saved_col_config is not None:
            # load from cache
            loaded_attr = saved_col_config['stat_attr']
            setaside_zero_v = loaded_attr.setaside_zero_
            log_scale_v = loaded_attr.log_scale_
            xtreme_method_v = loaded_attr.xtreme_method_
            bins_v = loaded_attr.bins_
        else:
            # initial value
            setaside_zero_v = False
            log_scale_v = False
            xtreme_method_v = 'iqr'
            bins_v = 100
        
        with feature_stat_option_cols[1]:
            cols_num_ = st.columns(2)
            with cols_num_[0]:
                num_bins_inp_ = st.number_input(
                    label = '# of Bins for histogram',
                    min_value = 2,
                    max_value = 500,
                    value = bins_v,
                    step = 1
                )
                setaside_zero_check_ = st.checkbox(
                    label = 'Set aside Zeros',
                    value = setaside_zero_v,
                )
            with cols_num_[1]:
                xtreme_options = [None, 'iqr', 'quantile']
                xtreme_method_select_ = st.selectbox(
                    label = 'Xtreme Value Detection',
                    options = xtreme_options,
                    index = xtreme_options.index(xtreme_method_v)
                )
                logscale_check_ = st.checkbox(
                    label = 'Add Log Transform',
                    value = log_scale_v,
                )
                
        # preprocess widges
        # important! must be same type to load back params, or will load to wrong class
        if guessed_dtype_ == dtype_choice_ and saved_col_config is not None:
            # load from cache
            loaded_preproc = saved_col_config['preproc']
            impute_v = loaded_preproc.impute
            normalize_v = loaded_preproc.normalize
            standardize_strategy_v = loaded_preproc.standardize_strategy
        else:
            impute_v = True
            normalize_v = False
            standardize_strategy_v = 'robust'
            
        btn_cols = st.columns(3)
        with btn_cols[0]:
            with st.popover("Preprocess Instruction"):
                impute_toggle_ = st.toggle(
                    label = 'Apply Imputation',
                    value = impute_v,
                )
                normalize_toggle_ = st.toggle(
                    label = 'Apply Normalization',
                    value = normalize_v,
                )
                standardize_strategy_ = st.selectbox(
                    label = 'Standardization Strategy',
                    options = ['robust', 'standard', 'maxabs'],
                    index = ['robust', 'standard', 'maxabs'].index(standardize_strategy_v)
                )
        
        # preprocess options    
        attr = NumericStatAttr(
            setaside_zero_ = setaside_zero_check_,
            log_scale_ = logscale_check_,
            xtreme_method_ = xtreme_method_select_,
            bins_ = num_bins_inp_
        )
        preproc = NumericFeaturePreproc(
            impute = impute_toggle_,
            normalize = normalize_toggle_,
            standardize_strategy = standardize_strategy_
        )
        
        with btn_cols[1]:
            st.button(
                label = 'Save/Update',
                type = 'primary',
                on_click = save_update_column_profile,
                kwargs = dict(
                    col = selected_feature_,
                    dtype = dtype_choice_,
                    stat_attr = attr,
                    preproc = preproc
                )
            )
        with btn_cols[2]:
            st.button(
                label = 'Remove',
                type = 'secondary',
                on_click = remove_profile,
                kwargs = dict(
                    col = selected_feature_,
                )
            )
        
        # display and info section (no data saving)
        summary_ = get_or_train_numeric_fsummary(
            data_config = st.session_state['data_config'],
            col = selected_feature_,
            attr = attr
        )
        
        # plot
        num_tabs = st.tabs(['Composit Donut', 'Histogram', 'Statistics', 'Xtremes'])
        plots = get_numeric_plots(summary_)
        with num_tabs[0]:
            st.bokeh_chart(
                figure = plots['donut'], 
                use_container_width = False
            )
        with num_tabs[1]:
            st.bokeh_chart(
                figure = plots['hist'], 
                use_container_width = False
            )  
        with num_tabs[2]:
            col_nums_stat_tb = st.columns(2)
            with col_nums_stat_tb[0]:
                st.bokeh_chart(
                    figure = plots['quant'], 
                    use_container_width = False
                )
            with col_nums_stat_tb[1]:
                st.bokeh_chart(
                    figure = plots['desc'], 
                    use_container_width = False
                )
        with num_tabs[3]:
            if plots.get('xtreme') is not None:
                st.bokeh_chart(
                    figure = plots['xtreme'], 
                    use_container_width = False
                )
    
    
    eda_model_id = st.text_input(
        label = "EDA Model Id",
        value = st.session_state['selected_eda_model_id'] or "",
    )
    if eda_model_id in get_data_manager().get_existing_eda_model_ids():
       st.warning("EDA Model Id already exist, will overwrite!")
    
    
    io_cols = st.columns(3)
    with io_cols[0]:
        st.button(
            label = 'Train and Save',
            type = 'primary',
            on_click = train_and_save_eda_model,
            kwargs = dict(
                eda_model_id = eda_model_id,
            )
        )
    with io_cols[1]:
        if 'tpm_list_downloadable' in st.session_state:
            st.download_button(
                label = "Download JSON",
                file_name = "feature_preprocess_model.json",
                mime = "application/json",
                data = json.dumps(
                    st.session_state['tpm_list_downloadable'], 
                    indent = 4
                ),
            )   
    with io_cols[2]:
        
        if 'tpm_list_downloadable' in st.session_state:
            st.download_button(
                label = "Download HTML",
                file_name = "feature_standalone_profile.html",
                mime = "application/html",
                data = plot_html(),
            )

    
    with st.expander(label = 'Shopping Cart', expanded = True):
        st.json(serialize_profile())
    