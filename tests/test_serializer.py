import importlib
import logging
import pathlib
import tempfile
import uuid
from collections import namedtuple

import hypothesis.strategies as st

logger = logging.getLogger(__name__)

_SUPPORTED_SERIALIZERS = [
    ("pickle", "b"),
    ("dill", "b"),
    ("autora.serializer.yaml_", ""),
]
_SERIALIZER_DEF = namedtuple("_SERIALIZER_DEF", ["name", "module", "file_type"])
_AVAILABLE_SERIALIZERS = []

for module_name, file_type in _SUPPORTED_SERIALIZERS:
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        logger.info(f"serializer {module} not available")
        continue
    _AVAILABLE_SERIALIZERS.append(_SERIALIZER_DEF(module_name, module, file_type))

AVAILABLE_SERIALIZERS = st.sampled_from(_AVAILABLE_SERIALIZERS)


@st.composite
def serializer_loads_dumps_strategy(draw):
    serializer = draw(AVAILABLE_SERIALIZERS)
    loads, dumps = serializer.module.loads, serializer.module.dumps
    return loads, dumps


@st.composite
def serializer_dump_load_string_strategy(draw):
    """Strategy returns a function which dumps an object and reloads it via a bytestream."""
    serializer = draw(AVAILABLE_SERIALIZERS)
    loads, dumps = serializer.module.loads, serializer.module.dumps

    def _load_dump_via_string(o):
        logger.info(f"load dump via string using {serializer.module=}")
        return loads(dumps(o))

    return _load_dump_via_string


@st.composite
def serializer_dump_load_binary_file_strategy(draw):
    """Strategy returns a function which dumps an object reloads it via a temporary binary file."""
    serializer = draw(AVAILABLE_SERIALIZERS)
    load, dump = serializer.module.load, serializer.module.dump

    def _load_dump_via_disk(o):
        logger.info(f"load dump via disk using {serializer.module=}")
        with tempfile.TemporaryDirectory() as tempdir:
            filename = str(uuid.uuid1())
            with open(pathlib.Path(tempdir, filename), f"w{serializer.file_mode}") as f:
                dump(o, f)
            with open(pathlib.Path(tempdir, filename), f"r{serializer.file_mode}") as f:
                o_loaded = load(f)
        return o_loaded

    return _load_dump_via_disk


@st.composite
def serializer_dump_load_strategy(draw):
    """Strategy returns a function which dumps an object and reloads it via a supported method."""
    _dump_load = draw(
        st.one_of(
            serializer_dump_load_string_strategy(),
            serializer_dump_load_binary_file_strategy(),
        )
    )
    return _dump_load


if __name__ == "__main__":
    o = list("abcde")
    loader_dumper_disk = serializer_dump_load_strategy().example()
    o_loaded = loader_dumper_disk(o)
    print(o, o_loaded)
