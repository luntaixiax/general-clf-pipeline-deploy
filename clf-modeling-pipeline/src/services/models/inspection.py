# model parameter and config
# model performance
# model comparison
# model explain (shap)
import pandas as pd
from src.services.models.registry import load_cp_model
from src.services.models.training_data import fetch_train_data

def get_coeff_impl(model_id: str) -> pd.Series:
    model = load_cp_model(model_id)
    try:
        r = model.getCoeffImp()
    except TypeError as e:
        r = pd.Series()
    return r

def test_models_score(model_id: str, data_id: str, use_train: bool = False, 
            use_calibrated_pred: bool = True, metric: str = 'balanced_accuracy') -> pd.DataFrame:
    
    # get model
    model = load_cp_model(model_id)
    le = model.getLabelEncoder()
    
    X_train, y_train, X_test, y_test = fetch_train_data(data_id=data_id)
    # get actual / pred data
    if use_train:
        y_pred = model.inference(X_train.to_pandas())
        y_true = le.transform(y_train.to_pandas())
    else:
        y_pred = model.inference(X_test.to_pandas())
        y_true = le.transform(y_test.to_pandas())
        
    # get uncalibrated or calibrated score
    if use_calibrated_pred:
        
        
    
    