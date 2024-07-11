from typing import List
import yaml
from dataclasses import dataclass
from dacite import from_dict
from functools import lru_cache

@lru_cache()
def get_settings(path: str = "src/configs/settings.yml") -> dict:
    with open(path, 'r') as obj:
        return yaml.safe_load(obj)

@dataclass
class ENTITIES:
    entity_key: str
    dt_key: str
    
    @property
    def pk(self) -> List[str]:
        return [self.entity_key, self.dt_key]
    
@dataclass
class TARGETS:
    target_key: str
    num_cls: int = 2
    
def get_entity_config() -> ENTITIES:
    return from_dict(
        data_class=ENTITIES,
        data = get_settings().get('entities', {})
    )
    
def get_target_config() -> TARGETS:
    return from_dict(
        data_class=TARGETS,
        data = get_settings().get('targets', {})
    )
    
ENTITY_CFG = get_entity_config()
TARGET_CFG = get_target_config()
