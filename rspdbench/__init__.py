"""RSPDBench: physically grounded remote-sensing product degradations for EO models."""
from .conditions import Condition, all_conditions, conditions_for, get_condition
from .contracts import CONTRACTS, MEUROSAT_STORAGE_ORDER, BandContract, canonical_band_id, get_contract
from .metrics import compound_excess, relative_drop
from .operators import degrade

__version__ = "1.0.1"
__all__ = [
    "Condition", "all_conditions", "conditions_for", "get_condition",
    "CONTRACTS", "MEUROSAT_STORAGE_ORDER", "BandContract", "canonical_band_id", "get_contract",
    "compound_excess", "relative_drop", "degrade",
]
