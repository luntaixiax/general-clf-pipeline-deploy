from datetime import date
from typing import List, Literal
from src.dao.dbapi import CONV_MODEL_DATA_REGISTRY

def get_train_data_ids() -> List[str]:
    """get a list of registered data id

    :return List[str]: list of data id
    """
    return CONV_MODEL_DATA_REGISTRY.get_existing_ids()

def create_train_data(data_id: str, use_snap_dts: List[date],
        sample_method: Literal['simple', 'stratify'] = 'stratify',
        split_method: Literal['simple', 'stratify', 'group', 'timeseries'] = 'group',
        sample_frac: float = 0.25, train_size: float = 0.8, random_seed: int = 0):
    """generate train and test set from database

    :param str data_id: the data id for the generated train/test dataset
    :param List[date] use_snap_dts: list of snap dates to use (sample from)
    :param str sample_method: one of 'simple', 'stratify', defaults to 'stratify'
    :param str split_method: one of 'simple', 'stratify', 'group', 'timeseries', defaults to 'group'
    :param float sample_frac: down sampling size, defaults to 0.25
    :param float train_size: train sample size as of total size, defaults to 0.8
    :param int random_seed: control randomness, defaults to 0
    """
    train_ds, test_ds = CONV_MODEL_DATA_REGISTRY.generate(
        use_snap_dts = use_snap_dts,
        sample_method = sample_method,
        split_method = split_method,
        sample_frac = sample_frac,
        train_size = train_size,
        random_seed = random_seed
    )
    CONV_MODEL_DATA_REGISTRY.register(
        data_id = data_id,
        train_ds = train_ds,
        test_ds = test_ds,
        replace = True
    )

def fetch_train_data(data_id: str) -> tuple:
    """fetch stored train data

    :param str data_id: the data id for the generated train/test dataset
    :return tuple: [X_train, y_train, X_test, y_test] in ibis dataset format
    """
    if data_id not in CONV_MODEL_DATA_REGISTRY.get_existing_ids():
        raise ValueError(f"{data_id} does not exist in data registry")
    
    X_train, y_train, X_test, y_test = CONV_MODEL_DATA_REGISTRY.fetch(
        data_id = data_id, 
        target_col = CONV_MODEL_DATA_REGISTRY.TARGET_COL
    )
    return X_train, y_train, X_test, y_test
    
    
def remove_train_data(data_id: str):
    """remove training data

    :param str data_id: the data id for the generated train/test dataset
    """
    CONV_MODEL_DATA_REGISTRY.remove(data_id=data_id)