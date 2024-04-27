from dataclasses import dataclass, field
from typing import Any, Literal
import optuna
from sklearn.base import clone, BaseEstimator, TransformerMixin

@dataclass
class HyperMode:
    hyper_mode: bool = False
    hyper_storage: optuna.storages.RDBStorage | None = None
    attrs: dict = field(default_factory=dict) 