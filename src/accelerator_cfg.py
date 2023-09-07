from enum import Enum


class AcceleratorType(Enum):
    Eyeriss = 1


class AcceleratorProfile:
    """General characteristics from the given accelerator
    """
    def __init__(self, accelerator_type):
        self.type = accelerator_type
        if accelerator_type == AcceleratorType.Eyeriss:
            self.width = 14
            self.height = 16
        else:
            raise NotImplementedError

