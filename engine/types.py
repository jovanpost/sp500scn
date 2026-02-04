# engine/types.py
from dataclasses import dataclass, field
from typing import Optional

@dataclass(frozen=True)
class SpikeDefinition:
    """
    Shared source of truth for what defines a 'Spike'.
    Used by both the Backtester/Engine and the UI Lab.
    """
    lookback_days: int = 20
    min_pct_up: float = 8.0
    volume_multiple: float = 2.0
    require_precursor: bool = True
    
    # Metadata for UI/Reporting
    label: Optional[str] = None
