import logging
from functools import wraps

_logger = logging.getLogger(__name__)


def deprecated_alias(f, alias_name):
    """Wrapper to make function aliases which print a warning that a name is an alias."""

    @wraps(f)
    def wrapper(*args, **kwds):
        _logger.warning(
            f"use {f.__name__} instead. " f"{alias_name} is a deprecated alias."
        )
        return f(*args, **kwds)

    return wrapper
