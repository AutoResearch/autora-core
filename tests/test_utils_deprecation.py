from hypothesis import given
from hypothesis.strategies import text

from autora.utils.deprecation import deprecated_alias


@given(text())
def test_deprecated_function_all_strings_are_allowed_for_names(name):

    c = ""  # the callback argument

    def update_c(s):  # the callback function
        nonlocal c
        c = s

    deprecated_alias(lambda: None, name, callback=update_c)()

    assert name in c
