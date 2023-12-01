import logging
import pathlib
import tempfile
import uuid
from functools import partial

import hypothesis.strategies as st

from autora.serializer import SupportedSerializer, load_serializer

logger = logging.getLogger(__name__)


def _load_dump_via_string(o, module_name):
    s = load_serializer(module_name)
    return s.module.loads(s.module.dumps(o))


def _load_dump_via_disk(o, module_name):
    s = load_serializer(module_name)
    with tempfile.TemporaryDirectory() as tempdir:
        filename = str(uuid.uuid1())
        with open(pathlib.Path(tempdir, filename), f"w{s.file_mode}") as f:
            s.module.dump(o, f)
        with open(pathlib.Path(tempdir, filename), f"r{s.file_mode}") as f:
            o_loaded = s.module.load(f)
    return o_loaded


# Preload all the serializer modules to avoid delays when running each the first time (which could
# cause issues for hypothesis)
[load_serializer(s) for s in SupportedSerializer]

# Explicitly list all the strategies we use
load_dump_pickle_string = partial(_load_dump_via_string, module_name="pickle")
load_dump_pickle_disk = partial(_load_dump_via_disk, module_name="pickle")
load_dump_dill_string = partial(_load_dump_via_string, module_name="dill")
load_dump_dill_disk = partial(_load_dump_via_disk, module_name="dill")
load_dump_yaml_string = partial(_load_dump_via_string, module_name="yaml")
load_dump_yaml_disk = partial(_load_dump_via_disk, module_name="yaml")


serializer_dump_load_strategy = st.sampled_from(
    [
        load_dump_pickle_string,
        load_dump_dill_string,
        load_dump_yaml_string,
        load_dump_pickle_disk,
        load_dump_dill_disk,
        load_dump_yaml_disk,
    ]
)

if __name__ == "__main__":
    o = list("abcde")
    loader_dumper_disk = serializer_dump_load_strategy.example()
    o_loaded = loader_dumper_disk(o)
    print(o, o_loaded)
