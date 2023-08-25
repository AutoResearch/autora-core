import pathlib
import tempfile

from hypothesis import Verbosity, given, settings
from hypothesis import strategies as st

from autora.state import StandardState
from autora.workflow.__main__ import _dump_state, _load_state


@given(
    st.builds(StandardState, st.text(), st.text(), st.text(), st.lists(st.integers()))
)
@settings(verbosity=Verbosity.verbose)
def test_load_inverts_dump(s):
    with tempfile.TemporaryDirectory() as dir:
        path = pathlib.Path(dir, "x.dill")
        print(path, s)
        _dump_state(s, path)
        assert _load_state(path) == s
