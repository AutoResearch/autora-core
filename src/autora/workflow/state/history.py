import logging

from autora.state.history import History  # noqa: F401

_logger = logging.getLogger(__name__)

_logger.warning(
    "`autora.workflow.state.history` will be removed in a future release. "
    "Use `autora.state.history` instead."
)
