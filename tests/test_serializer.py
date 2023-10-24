import pathlib
import tempfile
import uuid

from hypothesis import Verbosity, given, settings
from hypothesis import strategies as st

from autora.serializer import SerializersSupported, dump_state, load_state
from autora.state import StandardState


@given(
    st.builds(StandardState, st.text(), st.text(), st.text(), st.lists(st.integers())),
    st.sampled_from(SerializersSupported),
)
@settings(verbosity=Verbosity.verbose)
def test_load_inverts_dump(s, serializer):
    """Test that each serializer can be used to serialize and deserialize a state object."""
    with tempfile.TemporaryDirectory() as dir:
        path = pathlib.Path(dir, f"{str(uuid.uuid4())}")
        print(path, s)

        dump_state(s, path, dumper=serializer)
        assert load_state(path, loader=serializer) == s
