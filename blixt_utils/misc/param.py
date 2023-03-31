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

    def __repr__(self):
        # if len(self.desc) == 0:
        #     return '{} = {} [{}]'.format(self.name, self.value, self.unit)
        # else:
        #     return '{} = {} [{}], {}'.format(self.name, self.value, self.unit, self.desc)
        return '{}'.format(self.value)
