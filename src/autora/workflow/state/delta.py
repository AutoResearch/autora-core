import logging

from autora.state.delta import Delta, Result, State, wrap_to_use_state  # noqa: F401

_logger = logging.getLogger(__name__)

_logger.warning(
    "`autora.workflow.state.delta` will be removed in a future release. "
    "Use `autora.state.delta` instead."
)
