"""Load the frozen numerical risk predictor without unpickling Python objects."""
from pathlib import Path
import json
import numpy as np
import lightgbm as lgb

class FrozenRiskPredictor:
    def __init__(self,model_directory):
        path=Path(model_directory)
        self.schema=json.loads((path/'risk_preprocessing.json').read_text())
        self.booster=lgb.Booster(model_file=str(path/'risk_2026_08_lightgbm.txt'))

    def predict(self,frame):
        s=self.schema
        missing=set(s['feature_columns'])-set(frame.columns)
        if missing:raise ValueError(f'Missing feature columns: {sorted(missing)}')
        # Same order as ColumnTransformer: categorical one-hot then standardized
        # numeric fields. Unknown categories receive all-zero one-hot encoding.
        ids=frame[s['categorical_column']].to_numpy()
        categorical=(ids[:,None]==np.asarray(s['categories'])[None,:]).astype(float)
        numeric=frame[s['numeric_columns']].to_numpy(dtype=float)
        numeric=(numeric-np.asarray(s['mean']))/np.asarray(s['scale'])
        if not np.isfinite(numeric).all():raise ValueError('Core inputs must be finite')
        matrix=np.concatenate([categorical,numeric],axis=1)
        return np.maximum(s['prediction_clip_min'],self.booster.predict(matrix)+s['base_score'])
