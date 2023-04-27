from autora.synthetic.inventory import SyntheticExperimentCollection, retrieve, register
from autora.variable import VariableCollection


def test_model_registration_retrieval():
    # We can register a model and retrieve it
    register("empty", lambda: SyntheticExperimentCollection())
    empty = retrieve("empty")
    assert empty.name is None

    # We can register another model and retrieve it as well
    register(
        "only_metadata",
        lambda: SyntheticExperimentCollection(metadata=VariableCollection()),
    )
    only_metadata = retrieve("only_metadata")
    assert only_metadata.metadata is not None

    # We can still retrieve the first model, and it is equal to the first version
    empty_copy = retrieve("empty")
    assert empty_copy == empty
