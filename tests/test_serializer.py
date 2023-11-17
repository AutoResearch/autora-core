import pickle

from hypothesis import strategies as st

SUPPORTED_SERIALIZERS = st.sampled_from([(pickle.loads, pickle.dumps)])
