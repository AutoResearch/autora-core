import pathlib
import tempfile
import uuid

from hypothesis import Verbosity, given, settings
from hypothesis import strategies as st

from autora.state import StandardState
from autora.workflow.serializer import Supported, _load_state, dump_state


@given(
    st.builds(StandardState, st.text(), st.text(), st.text(), st.lists(st.integers())),
    st.sampled_from(Supported),
)
@settings(verbosity=Verbosity.verbose)
def test_load_inverts_dump(s, serializer):
    with tempfile.TemporaryDirectory() as dir:
        path = pathlib.Path(dir, f"{str(uuid.uuid4())}.{serializer}")
        print(path, s)

        dump_state(s, path, dumper=serializer)
        assert _load_state(path, loader=serializer) == s
