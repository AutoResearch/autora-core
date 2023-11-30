import logging
import pathlib
import tempfile
from typing import Optional

import numpy as np
from hypothesis import Verbosity, given, settings
from hypothesis import strategies as st

from autora.serializer import SupportedSerializer, load_state
from autora.state import StandardState
from autora.workflow.__main__ import main

_logger = logging.getLogger(__name__)

example_workflow_library_module = st.sampled_from(["_example_workflow_library"])


def validate_model(state: Optional[StandardState]):
    assert state is not None

    assert state.conditions is not None
    assert len(state.conditions) == 100

    assert state.experiment_data is not None
    assert len(state.experiment_data) == 100

    assert state.models[-1] is not None
    assert np.allclose(state.models[-1].coef_, [[2.0]])
    assert np.allclose(state.models[-1].intercept_, [[0.5]])


@given(example_workflow_library_module)
def test_e2e_nominal(workflow_library_module):
    """Test a basic standard chain of CLI calls using the default serializer.

    Equivalent to:
    $ python -m autora.workflow test_workflow.initial_state --out-path start
    $ python -m autora.workflow test_workflow.experimentalist --in-path start --out-path conditions
    $ python -m autora.workflow test_workflow.experiment_runner --in-path conditions --out-path data
    $ python -m autora.workflow test_workflow.theorist --in-path data --out-path theory
    """

    with tempfile.TemporaryDirectory() as d:
        main(
            f"{workflow_library_module}.initial_state",
            out_path=pathlib.Path(d, "start"),
        )
        main(
            f"{workflow_library_module}.experimentalist",
            in_path=pathlib.Path(d, "start"),
            out_path=pathlib.Path(d, "conditions"),
        )
        main(
            f"{workflow_library_module}.experiment_runner",
            in_path=pathlib.Path(d, "conditions"),
            out_path=pathlib.Path(d, "data"),
        )
        main(
            f"{workflow_library_module}.theorist",
            in_path=pathlib.Path(d, "data"),
            out_path=pathlib.Path(d, "theory"),
        )

        final_state = load_state(pathlib.Path(d, "theory"))
        validate_model(final_state)


@given(
    example_workflow_library_module,
    st.sampled_from(SupportedSerializer),
    st.booleans(),
    st.booleans(),
)
@settings(verbosity=Verbosity.verbose, deadline=1000)
def test_e2e_serializers(workflow_library_module, serializer, verbose, debug):
    """Test a basic standard chain of CLI calls using a single serializer."""

    common_settings = dict(
        in_loader=serializer, out_dumper=serializer, verbose=verbose, debug=debug
    )

    with tempfile.TemporaryDirectory() as d:
        main(
            f"{workflow_library_module}.initial_state",
            out_path=pathlib.Path(d, "start"),
            **common_settings,
        )
        main(
            f"{workflow_library_module}.experimentalist",
            in_path=pathlib.Path(d, "start"),
            out_path=pathlib.Path(d, "conditions"),
            **common_settings,
        )
        main(
            f"{workflow_library_module}.experiment_runner",
            in_path=pathlib.Path(d, "conditions"),
            out_path=pathlib.Path(d, "data"),
            **common_settings,
        )
        main(
            f"{workflow_library_module}.theorist",
            in_path=pathlib.Path(d, "data"),
            out_path=pathlib.Path(d, "theory"),
            **common_settings,
        )

        final_state: StandardState = load_state(
            pathlib.Path(d, "theory"), loader=serializer
        )
        validate_model(final_state)


@given(
    example_workflow_library_module,
    st.sampled_from(SupportedSerializer),
    st.sampled_from(SupportedSerializer),
    st.sampled_from(SupportedSerializer),
    st.sampled_from(SupportedSerializer),
    st.booleans(),
    st.booleans(),
)
@settings(verbosity=Verbosity.verbose, deadline=1000)
def test_e2e_valid_serializer_mix(
    workflow_library_module,
    initial_serializer,
    experimental_serializer,
    experiment_runner_serializer,
    theorist_serializer,
    verbose,
    debug,
):
    """Test a basic standard chain of CLI calls using a mix of serializers."""

    common_settings = dict(verbose=verbose, debug=debug)

    with tempfile.TemporaryDirectory() as d:
        main(
            f"{workflow_library_module}.initial_state",
            out_path=pathlib.Path(d, "start"),
            out_dumper=initial_serializer,
            **common_settings,
        )
        main(
            f"{workflow_library_module}.experimentalist",
            in_path=pathlib.Path(d, "start"),
            out_path=pathlib.Path(d, "conditions"),
            in_loader=initial_serializer,
            out_dumper=experimental_serializer,
            **common_settings,
        )
        main(
            f"{workflow_library_module}.experiment_runner",
            in_path=pathlib.Path(d, "conditions"),
            out_path=pathlib.Path(d, "data"),
            in_loader=experimental_serializer,
            out_dumper=experiment_runner_serializer,
            **common_settings,
        )
        main(
            f"{workflow_library_module}.theorist",
            in_path=pathlib.Path(d, "data"),
            out_path=pathlib.Path(d, "theory"),
            in_loader=experiment_runner_serializer,
            out_dumper=theorist_serializer,
            **common_settings,
        )

        final_state: StandardState = load_state(
            pathlib.Path(d, "theory"), loader=theorist_serializer
        )
        validate_model(final_state)
