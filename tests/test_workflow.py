import logging
import pathlib
import tempfile
from typing import Optional

import numpy as np
import pandas as pd
from hypothesis import Verbosity, given, settings
from hypothesis import strategies as st
from sklearn.linear_model import LinearRegression

from autora.experimentalist.grid import grid_pool
from autora.serializer import SerializersSupported, load_state
from autora.state import StandardState, State, estimator_on_state, on_state
from autora.variable import Variable, VariableCollection
from autora.workflow.__main__ import main

_logger = logging.getLogger(__name__)


def initial_state(_):
    state = StandardState(
        variables=VariableCollection(
            independent_variables=[Variable(name="x", allowed_values=range(100))],
            dependent_variables=[Variable(name="y")],
            covariates=[],
        ),
        conditions=None,
        experiment_data=pd.DataFrame({"x": [], "y": []}),
        models=[],
    )
    return state


experimentalist = on_state(grid_pool, output=["conditions"])

experiment_runner = on_state(
    lambda conditions: conditions.assign(y=2 * conditions["x"] + 0.5),
    output=["experiment_data"],
)

theorist = estimator_on_state(LinearRegression(fit_intercept=True))


def validate_model(state: Optional[State]):
    assert state is not None

    assert state.conditions is not None
    assert len(state.conditions) == 100

    assert state.experiment_data is not None
    assert len(state.experiment_data) == 100

    assert state.model is not None
    assert np.allclose(state.model.coef_, [[2.0]])
    assert np.allclose(state.model.intercept_, [[0.5]])


def test_e2e_nominal():
    """Test a basic standard chain of CLI calls using the default serializer.

    Equivalent to:
    $ python -m autora.workflow test_workflow.initial_state --out-path start
    $ python -m autora.workflow test_workflow.experimentalist --in-path start --out-path conditions
    $ python -m autora.workflow test_workflow.experiment_runner --in-path conditions --out-path data
    $ python -m autora.workflow test_workflow.theorist --in-path data --out-path theory
    """

    with tempfile.TemporaryDirectory() as d:
        main(
            "test_workflow.initial_state",
            out_path=pathlib.Path(d, "start"),
        )
        main(
            "test_workflow.experimentalist",
            in_path=pathlib.Path(d, "start"),
            out_path=pathlib.Path(d, "conditions"),
        )
        main(
            "test_workflow.experiment_runner",
            in_path=pathlib.Path(d, "conditions"),
            out_path=pathlib.Path(d, "data"),
        )
        main(
            "test_workflow.theorist",
            in_path=pathlib.Path(d, "data"),
            out_path=pathlib.Path(d, "theory"),
        )

        final_state = load_state(pathlib.Path(d, "theory"))
        validate_model(final_state)


@given(st.sampled_from(SerializersSupported), st.booleans(), st.booleans())
@settings(verbosity=Verbosity.verbose, deadline=500)
def test_e2e_serializers(serializer, verbose, debug):
    """Test a basic standard chain of CLI calls using a single serializer."""

    common_settings = dict(
        in_loader=serializer, out_dumper=serializer, verbose=verbose, debug=debug
    )

    with tempfile.TemporaryDirectory() as d:
        main(
            "test_workflow.initial_state",
            out_path=pathlib.Path(d, "start"),
            **common_settings
        )
        main(
            "test_workflow.experimentalist",
            in_path=pathlib.Path(d, "start"),
            out_path=pathlib.Path(d, "conditions"),
            **common_settings
        )
        main(
            "test_workflow.experiment_runner",
            in_path=pathlib.Path(d, "conditions"),
            out_path=pathlib.Path(d, "data"),
            **common_settings
        )
        main(
            "test_workflow.theorist",
            in_path=pathlib.Path(d, "data"),
            out_path=pathlib.Path(d, "theory"),
            **common_settings
        )

        final_state: StandardState = load_state(
            pathlib.Path(d, "theory"), loader=serializer
        )
        validate_model(final_state)


@given(
    st.sampled_from(SerializersSupported),
    st.sampled_from(SerializersSupported),
    st.sampled_from(SerializersSupported),
    st.sampled_from(SerializersSupported),
    st.booleans(),
    st.booleans(),
)
@settings(verbosity=Verbosity.verbose, deadline=500)
def test_e2e_valid_serializer_mix(
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
            "test_workflow.initial_state",
            out_path=pathlib.Path(d, "start"),
            out_dumper=initial_serializer,
            **common_settings
        )
        main(
            "test_workflow.experimentalist",
            in_path=pathlib.Path(d, "start"),
            out_path=pathlib.Path(d, "conditions"),
            in_loader=initial_serializer,
            out_dumper=experimental_serializer,
            **common_settings
        )
        main(
            "test_workflow.experiment_runner",
            in_path=pathlib.Path(d, "conditions"),
            out_path=pathlib.Path(d, "data"),
            in_loader=experimental_serializer,
            out_dumper=experiment_runner_serializer,
            **common_settings
        )
        main(
            "test_workflow.theorist",
            in_path=pathlib.Path(d, "data"),
            out_path=pathlib.Path(d, "theory"),
            in_loader=experiment_runner_serializer,
            out_dumper=theorist_serializer,
            **common_settings
        )

        final_state: StandardState = load_state(
            pathlib.Path(d, "theory"), loader=theorist_serializer
        )
        validate_model(final_state)
