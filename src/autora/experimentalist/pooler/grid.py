import logging

from autora.experimentalist.grid_ import grid_pool as grid_pool_
from autora.utils.deprecation import deprecate

_logger = logging.getLogger(__name__)

grid_pool = deprecate(
    grid_pool_,
    "`from autora.experimentalist.pooler.grid import grid_pool` "
    "will be deprecated in future. Instead, use"
    "`from autora.experimentalist.grid import grid_pool`",
    callback=_logger.info,
)
