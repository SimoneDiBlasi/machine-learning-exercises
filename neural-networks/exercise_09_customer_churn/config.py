from pathlib import Path


EXERCISE_DIR = Path(__file__).resolve().parent
NEURAL_NETWORKS_DIR = EXERCISE_DIR.parent


# Dataset
DATASET_PATH = (
    NEURAL_NETWORKS_DIR
    / "datasets"
    / "Telco-Customer-Churn.csv"
)


# Output directories
MODELS_DIR = EXERCISE_DIR / "models"
RESULTS_DIR = EXERCISE_DIR / "results"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# Saved model artifacts
MODEL_PATH = MODELS_DIR / "customer_churn_model.keras"

PREPROCESSOR_PATH = (
    MODELS_DIR
    / "customer_churn_preprocessor.joblib"
)

CONFIG_PATH = (
    MODELS_DIR
    / "customer_churn_config.json"
)


# Visualization
ARCHITECTURE_IMAGE_PATH = (
    RESULTS_DIR
    / "customer_churn_model_architecture.png"
)


# Dataset configuration
RANDOM_STATE = 42
TEST_SIZE = 0.30
VALIDATION_SIZE = 0.50


# Training configuration
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.001


# Model configuration
DROPOUT_RATE_1 = 0.30
DROPOUT_RATE_2 = 0.20


# Default classification threshold
DEFAULT_THRESHOLD = 0.50