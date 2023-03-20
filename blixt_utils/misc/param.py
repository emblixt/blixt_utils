# Simple class for holding parameters in blixt_rp

from dataclasses import dataclass

@dataclass
class Param:
    """
    Data class to hold the different parameters used in blixt_rp
    """
    name: str
    value: float
    unit: str
    desc: str
