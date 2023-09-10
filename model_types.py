
# John Lambert

from enum import Enum

class AutoNumber(Enum):
    def __new__(cls):
        value = len(cls.__members__) + 1
        obj = object.__new__(cls)
        obj._value_ = value
        return obj

class ModelType(AutoNumber):
    DROPOUT_FN_OF_XSTAR = ()
    DROPOUT_RANDOM_GAUSSIAN_NOISE = ()
    DROPOUT_INFORMATION = ()
