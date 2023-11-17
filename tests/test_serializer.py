import importlib
import logging

import hypothesis.strategies as st

logger = logging.getLogger(__name__)

_SUPPORTED_SERIALIZERS = ["pickle", "dill"]
_AVAILABLE_SERIALIZERS = []

for module_name in _SUPPORTED_SERIALIZERS:
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        logger.info(f"serializer {module} not available")
        continue
    _AVAILABLE_SERIALIZERS.append(module)

AVAILABLE_SERIALIZERS = st.sampled_from(_AVAILABLE_SERIALIZERS)


@st.composite
def serializer_loads_dumps_strategy(draw):
    module = draw(AVAILABLE_SERIALIZERS)
    loads, dumps = module.loads, module.dumps
    return loads, dumps


@st.composite
def serializer_dump_load_strategy(draw):
    module = draw(AVAILABLE_SERIALIZERS)
    loads, dumps = module.loads, module.dumps
    return lambda x: loads(dumps(x))
