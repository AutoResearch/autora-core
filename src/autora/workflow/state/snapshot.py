import logging

from autora.state.snapshot import Snapshot  # noqa: F401

_logger = logging.getLogger(__name__)

_logger.warning(
    "`autora.workflow.state.snapshot` will be removed in a future release. "
    "Use `autora.state.snapshot` instead."
)
