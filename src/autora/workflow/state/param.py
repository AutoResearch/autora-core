import logging

from autora.state.param import resolve_state_params  # noqa: F401

_logger = logging.getLogger(__name__)

_logger.warning(
    "`autora.workflow.state.param` will be removed in a future release. "
    "Use `autora.state.param` instead."
)
