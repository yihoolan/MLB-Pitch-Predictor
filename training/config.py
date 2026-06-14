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

### Data split constants (≈70/15/15 temporal split across years)
### Train:  2023 Apr–Sep  +  2024 Apr–Jul
### Val:    2024 Aug–Sep
### Test:   2025 Apr–May
TRAIN_END_MONTH: dict[int, int] = {2023: 9, 2024: 7}
VAL_YEAR = 2024
VAL_MONTHS = (8, 9)
TEST_YEAR = 2025
TEST_MONTHS = (4, 5)
