"""Central configuration: paths, constants, reproducibility."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "cache"
EXPERIMENTS_DIR = ROOT / "experiments"
MODELS_DIR = ROOT / "experiments" / "checkpoints"

DATASET_URL = "https://github.com/sowide/Multi-modal-bankrutpcy/raw/main/dataset_paper.zip"

SEED = 42

# The 18 raw accounting items ($ millions), as named in the dataset columns.
RAW_ITEMS = [
    "current_assets", "total_assets", "cost_of_goods_sold", "total_long_term_debt",
    "depreciation_and_amortization", "ebit", "ebitda", "gross_profit", "inventory",
    "total_current_liabilities", "net_income", "retained_earnings", "total_receivables",
    "total_revenue", "market_value", "total_liabilities", "net_sales",
    "total_operating_expenses",
]

# Year prefixes in the financial CSVs. Verified empirically: '3_' is the most
# recent fiscal year t (failed firms' retained earnings/net income deteriorate
# monotonically toward '3_'), '1_' is t-2.
YEAR_PREFIXES = ["1", "2", "3"]  # chronological order: t-2, t-1, t

SPLITS = ["train", "validation", "test"]
TEXT_ITEMS = ["item_1", "item_7"]  # item_5 is nearly empty -> excluded

FINBERT_MODEL = "yiyanghkust/finbert-pretrain"
EMB_DIM = 768
MAX_CHUNKS_PER_ITEM = 8    # first N chunks per 10-K item (document order)
MAX_TEXT_TOKENS = 16       # per company: item_1 chunks + item_7 chunks, capped

# Model hyperparameters (shared defaults)
D_MODEL = 64
N_HEADS = 4
DROPOUT = 0.3
LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 64
MAX_EPOCHS = 60
PATIENCE = 8

for d in (RAW_DIR, PROCESSED_DIR, CACHE_DIR, EXPERIMENTS_DIR, MODELS_DIR):
    d.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = SEED):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(False)
    except ImportError:
        pass
