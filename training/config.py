from utils.feature_names import PITCH_TYPES

MLFLOW_EXPERIMENT = "pitch_type_lgbm"
REGISTERED_MODEL_NAME = "PitchTypeClassifier"
RANDOM_STATE = 42

LGBM_PARAMS: dict = {
    "objective": "multiclass",
    "num_class": len(PITCH_TYPES),
    "num_leaves": 63,
    "min_child_samples": 50,
    "learning_rate": 0.05,
    "verbose": -1,
    "seed": RANDOM_STATE,
}
N_ESTIMATORS = 1000
EARLY_STOPPING_ROUNDS = 50
