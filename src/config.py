import os
from pathlib import Path
from dataclasses import dataclass

# Base directory of the project (one level up from src)
BASE_DIR = Path(__file__).resolve().parent.parent

@dataclass
class Paths:
    """Directory paths for the project"""
    DATA_DIR: Path = BASE_DIR / "data"
    EMBEDDINGS_DIR: Path = BASE_DIR / "embeddings"
    MODELS_DIR: Path = BASE_DIR / "models"
    RESULTS_DIR: Path = BASE_DIR / "results"
    REPORTS_DIR: Path = BASE_DIR / "reports"

@dataclass
class PreprocessingConfig:
    """Hyperparameters for data preprocessing"""
    IMAGE_SIZE: tuple = (224, 224)  # Standard size for ResNet/EfficientNet
    BATCH_SIZE: int = 32
    NUM_WORKERS: int = 0
    # Standard ImageNet normalization values
    NORMALIZE_MEAN: tuple = (0.485, 0.456, 0.406)
    NORMALIZE_STD: tuple = (0.229, 0.224, 0.225)

@dataclass
class ModelConfig:
    """Configuration for embedding extraction and classifiers"""
    # Which pretrained CNN to use as the feature extractor
    FEATURE_EXTRACTOR_NAME: str = "resnet50"  # Options: resnet50, efficientnet_b0, etc.
    DEVICE: str = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
    RANDOM_SEED: int = 42

@dataclass
class AppConfig:
    """Configuration for the Streamlit App"""
    APP_TITLE: str = "DermAssist: Explainable Skin Lesion Classification"
    CONFIDENCE_THRESHOLD: float = 0.5

# Instantiate the configurations so they can be easily imported
paths = Paths()
preprocessing_config = PreprocessingConfig()
model_config = ModelConfig()
app_config = AppConfig()

# Create directories if they don't exist
for path in [paths.DATA_DIR, paths.EMBEDDINGS_DIR, paths.MODELS_DIR, paths.RESULTS_DIR, paths.REPORTS_DIR]:
    path.mkdir(parents=True, exist_ok=True)
