import logging
from datetime import date
from luntaiDs.ProviderTools.clickhouse.snap_struct import SnapshotDataManagerCHSQL
from luntaiDs.CommonTools.SnapStructure.dependency import _CurrentStream, _PastStream, _FutureStream
from src.pipeline.utils import SnapTableTransfomer
from src.pipeline.data_transform import Features
from src.services.models.registry import get_model_id_from_timetable, get_prod_model_id
from src.services.models.inference import predict_batch
from src.utils.settings import ENTITY_CFG

class Scoring(SnapTableTransfomer):
    dm = SnapshotDataManagerCHSQL(
        schema = 'TARGET', 
        table = 'SCORING', 
        snap_dt_key = ENTITY_CFG.dt_key
    )
    upstreams = [_CurrentStream(Features())]
    
    @classmethod
    def transform(cls, snap_dt: date):
        if model_id := get_model_id_from_timetable(snap_dt=snap_dt) \
                or get_prod_model_id() is None:
            raise ValueError("model id not defined in either time table or deployed as prod")
            
        df = Features.dm.read(snap_dt=snap_dt)
        predict_df = predict_batch(model_id=model_id, df=df)
        
        predict_df[ENTITY_CFG.pk] = df[ENTITY_CFG.pk]
        
        cls.dm.save(
            df = predict_df,
            snap_dt = snap_dt,
            overwrite = False
        )