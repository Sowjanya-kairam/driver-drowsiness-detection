from pathlib import Path

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
CLASS_NAMES = ["drowsy", "non_drowsy"]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "dataset"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports"
