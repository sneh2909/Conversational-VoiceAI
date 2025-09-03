from abc import ABCMeta


class SkenSingleton(ABCMeta):
    """This is a SkenSingleton metaclass for implementing SkenSingleton interfaces.
    author:andy
    Args:
        ABCMeta (_type_): _description_
    Returns:
        _type_: _description_
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(SkenSingleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]