"""Classes to represent cycle state $S$ as $S_n = S_{0} + \\sum_{i=1}^n \\Delta S_{i}$."""
from __future__ import annotations

import dataclasses
import inspect
import logging
import warnings
from collections import UserDict
from dataclasses import dataclass, fields, replace
from functools import singledispatch, wraps
from typing import Callable, Generic, List, Optional, Sequence, TypeVar

import numpy as np
import pandas as pd

_logger = logging.getLogger(__name__)
S = TypeVar("S")
T = TypeVar("T")


@dataclass(frozen=True)
class State:
    """
    Base object for dataclasses which use the Delta mechanism.

    Examples:
        >>> from dataclasses import dataclass, field
        >>> from typing import List, Optional

        We define a dataclass where each field (which is going to be delta-ed) has additional
        metadata "delta" which describes its delta behaviour.
        >>> @dataclass(frozen=True)
        ... class ListState(State):
        ...    l: List = field(default_factory=list, metadata={"delta": "extend"})
        ...    m: List = field(default_factory=list, metadata={"delta": "replace"})

        Now we instantiate the dataclass...
        >>> l = ListState(l=list("abc"), m=list("xyz"))
        >>> l
        ListState(l=['a', 'b', 'c'], m=['x', 'y', 'z'])

        ... and can add deltas to it. `l` will be extended:
        >>> l + Delta(l=list("def"))
        ListState(l=['a', 'b', 'c', 'd', 'e', 'f'], m=['x', 'y', 'z'])

        ... wheras `m` will be replaced:
        >>> l + Delta(m=list("uvw"))
        ListState(l=['a', 'b', 'c'], m=['u', 'v', 'w'])

        ... they can be chained:
        >>> l + Delta(l=list("def")) + Delta(m=list("uvw"))
        ListState(l=['a', 'b', 'c', 'd', 'e', 'f'], m=['u', 'v', 'w'])

        ... and we update multiple fields with one Delta:
        >>> l + Delta(l=list("ghi"), m=list("rst"))
        ListState(l=['a', 'b', 'c', 'g', 'h', 'i'], m=['r', 's', 't'])

        A non-existent field will be ignored:
        >>> l + Delta(o="not a field")
        ListState(l=['a', 'b', 'c'], m=['x', 'y', 'z'])

        ... but will trigger a warning:
        >>> with warnings.catch_warnings(record=True) as w:
        ...     _ = l + Delta(o="not a field")
        ...     print(w[0].message) # doctest: +NORMALIZE_WHITESPACE
        These fields: ['o'] could not be used to update ListState,
        which has these fields & aliases: ['l', 'm']

        We can also use the `.update` method to do the same thing:
        >>> l.update(l=list("ghi"), m=list("rst"))
        ListState(l=['a', 'b', 'c', 'g', 'h', 'i'], m=['r', 's', 't'])

        We can also define fields which `append` the last result:
        >>> @dataclass(frozen=True)
        ... class AppendState(State):
        ...    n: List = field(default_factory=list, metadata={"delta": "append"})

        >>> m = AppendState(n=list("ɑβɣ"))
        >>> m
        AppendState(n=['ɑ', 'β', 'ɣ'])

        `n` will be appended:
        >>> m + Delta(n="∂")
        AppendState(n=['ɑ', 'β', 'ɣ', '∂'])

        The metadata key "converter" is used to coerce types (inspired by
        [PEP 712](https://peps.python.org/pep-0712/)):
        >>> @dataclass(frozen=True)
        ... class CoerceStateList(State):
        ...    o: Optional[List] = field(default=None, metadata={"delta": "replace"})
        ...    p: List = field(default_factory=list, metadata={"delta": "replace",
        ...                                                    "converter": list})

        >>> r = CoerceStateList()

        If there is no `metadata["converter"]` set for a field, no coercion occurs
        >>> r + Delta(o="not a list")
        CoerceStateList(o='not a list', p=[])

        If there is a `metadata["converter"]` set for a field, the data are coerced:
        >>> r + Delta(p="not a list")
        CoerceStateList(o=None, p=['n', 'o', 't', ' ', 'a', ' ', 'l', 'i', 's', 't'])

        If the input data are of the correct type, they are returned unaltered:
        >>> r + Delta(p=["a", "list"])
        CoerceStateList(o=None, p=['a', 'list'])

        With a converter, inputs are converted to the type output by the converter:
        >>> import pandas as pd
        >>> @dataclass(frozen=True)
        ... class CoerceStateDataFrame(State):
        ...    q: pd.DataFrame = field(default_factory=pd.DataFrame,
        ...                            metadata={"delta": "replace",
        ...                                      "converter": pd.DataFrame})

        If the type is already correct, the object is passed to the converter,
        but should be returned unchanged:
        >>> s = CoerceStateDataFrame()
        >>> (s + Delta(q=pd.DataFrame([("a",1,"alpha"), ("b",2,"beta")], columns=list("xyz")))).q
           x  y      z
        0  a  1  alpha
        1  b  2   beta

        If the type is not correct, the object is converted if possible. For a dataframe,
        we can convert records:
        >>> (s + Delta(q=[("a",1,"alpha"), ("b",2,"beta")])).q
           0  1      2
        0  a  1  alpha
        1  b  2   beta

        ... or an array:
        >>> (s + Delta(q=np.linspace([1, 2], [10, 15], 3))).q
              0     1
        0   1.0   2.0
        1   5.5   8.5
        2  10.0  15.0

        ... or a dictionary:
        >>> (s + Delta(q={"a": [1,2,3], "b": [4,5,6]})).q
           a  b
        0  1  4
        1  2  5
        2  3  6

        ... or a list:
        >>> (s + Delta(q=[11, 12, 13])).q
            0
        0  11
        1  12
        2  13

        ... but not, for instance, a string:
        >>> (s + Delta(q="not compatible with pd.DataFrame")).q
        Traceback (most recent call last):
        ...
        ValueError: DataFrame constructor not properly called!

        Without a converter:
        >>> @dataclass(frozen=True)
        ... class CoerceStateDataFrameNoConverter(State):
        ...    r: pd.DataFrame = field(default_factory=pd.DataFrame, metadata={"delta": "replace"})

        ... there is no coercion – the object is passed unchanged
        >>> t = CoerceStateDataFrameNoConverter()
        >>> (t + Delta(r=np.linspace([1, 2], [10, 15], 3))).r
        array([[ 1. ,  2. ],
               [ 5.5,  8.5],
               [10. , 15. ]])


        A converter can cast from a DataFrame to a np.ndarray (with a single datatype),
        for instance:
        >>> import numpy as np
        >>> @dataclass(frozen=True)
        ... class CoerceStateArray(State):
        ...    r: Optional[np.ndarray] = field(default=None,
        ...                            metadata={"delta": "replace",
        ...                                      "converter": np.asarray})

        Here we pass a dataframe, but expect a numpy array:
        >>> (CoerceStateArray() + Delta(r=pd.DataFrame([("a",1), ("b",2)], columns=list("xy")))).r
        array([['a', 1],
               ['b', 2]], dtype=object)

        We can define aliases which can transform between different potential field
        names.

        >>> @dataclass(frozen=True)
        ... class FieldAliasState(State):
        ...     things: List[str] = field(
        ...     default_factory=list,
        ...     metadata={"delta": "extend",
        ...               "aliases": {"thing": lambda m: [m]}}
        ...     )

        In the "normal" case, the Delta object is expected to include a list of data in the
        correct format which is used to extend the object:
        >>> FieldAliasState(things=["0"]) + Delta(things=["1", "2"])
        FieldAliasState(things=['0', '1', '2'])

        However, say the standard return from a step in AER is a single `thing`, rather than a
        sequence of them:
        >>> FieldAliasState(things=["0"]) + Delta(thing="1")
        FieldAliasState(things=['0', '1'])


        If a cycle function relies on the existence of the `s.thing` as a property of your state
        `s`, rather than accessing `s.things[-1]`, then you could additionally define a `property`:

        >>> class FieldAliasStateWithProperty(FieldAliasState):  # inherit from FieldAliasState
        ...     @property
        ...     def thing(self):
        ...         return self.things[-1]

        Now you can access both `s.things` and `s.thing` as required by your code. The State only
        shows `things` in the string representation...
        >>> s = FieldAliasStateWithProperty(things=["0"]) + Delta(thing="1")
        >>> s
        FieldAliasStateWithProperty(things=['0', '1'])

        ... and exposes `things` as an attribute:
        >>> s.things
        ['0', '1']

        ... but also exposes `thing`, always returning the last value.
        >>> s.thing
        '1'




    """

    def __add__(self, other: Delta):
        updates = dict()
        other_fields_unused = list(other.keys())
        for self_field in fields(self):

            other_value, key = _get_value(self_field, other)
            if other_value is None:
                continue
            other_fields_unused.remove(key)

            self_field_key = self_field.name
            self_value = getattr(self, self_field_key)
            delta_behavior = self_field.metadata["delta"]

            if (constructor := self_field.metadata.get("converter", None)) is not None:
                coerced_other_value = constructor(other_value)
            else:
                coerced_other_value = other_value

            if delta_behavior == "extend":
                extended_value = extend(self_value, coerced_other_value)
                updates[self_field_key] = extended_value
            elif delta_behavior == "append":
                appended_value = append(self_value, coerced_other_value)
                updates[self_field_key] = appended_value
            elif delta_behavior == "replace":
                updates[self_field_key] = coerced_other_value
            else:
                raise NotImplementedError(
                    "delta_behaviour=`%s` not implemented" % (delta_behavior)
                )

        if len(other_fields_unused) > 0:
            warnings.warn(
                "These fields: %s could not be used to update %s, "
                "which has these fields & aliases: %s"
                % (
                    other_fields_unused,
                    type(self).__name__,
                    _get_field_names_and_aliases(self),
                ),
            )

        new = replace(self, **updates)
        return new

    def update(self, **kwargs):
        return self + Delta(**kwargs)


def _get_value(f, other: Delta):
    """
    Given a `State`'s `dataclasses.field` f, get a value from `other` and report its name.

    Returns: a tuple (the value, the key associated with that value)

    Examples:
        >>> from dataclasses import field, dataclass, fields
        >>> @dataclass
        ... class Example:
        ...     a: int = field()  # base case
        ...     b: List[int] = field(metadata={"aliases": {"ba": lambda b: [b]}})  # Single alias
        ...     c: List[int] = field(metadata={"aliases": {
        ...                                         "ca": lambda x: x,   # pass the value unchanged
        ...                                         "cb": lambda x: [x]  # wrap the value in a list
        ...        }})  # Multiple alias

        For a field with no aliases, we retrieve values with the base name:
        >>> f_a = fields(Example)[0]
        >>> _get_value(f_a, Delta(a=1))
        (1, 'a')

        ... and only the base name:
        >>> print(_get_value(f_a, Delta(b=2)))  # no match for b
        (None, None)

        Any other names are unimportant:
        >>> _get_value(f_a, Delta(b=2, a=1))
        (1, 'a')

        For fields with an alias, we retrieve values with the base name:
        >>> f_b = fields(Example)[1]
        >>> _get_value(f_b, Delta(b=[2]))
        ([2], 'b')

        ... or for the alias name, transformed by the alias lambda function:
        >>> _get_value(f_b, Delta(ba=21))
        ([21], 'ba')

        We preferentially get the base name, and then any aliases:
        >>> _get_value(f_b, Delta(b=2, ba=21))
        (2, 'b')

        ... , regardless of their order in the `Delta` object:
        >>> _get_value(f_b, Delta(ba=21, b=2))
        (2, 'b')

        Other names are ignored:
        >>> _get_value(f_b, Delta(a=1))
        (None, None)

        and the order of other names is unimportant:
        >>> _get_value(f_b, Delta(a=1, b=2))
        (2, 'b')

        For fields with multiple aliases, we retrieve values with the base name:
        >>> f_c = fields(Example)[2]
        >>> _get_value(f_c, Delta(c=[3]))
        ([3], 'c')

        ... for any alias:
        >>> _get_value(f_c, Delta(ca=31))
        (31, 'ca')

        ... transformed by the alias lambda function :
        >>> _get_value(f_c, Delta(cb=32))
        ([32], 'cb')

        ... and ignoring any other names:
        >>> print(_get_value(f_c, Delta(a=1)))
        (None, None)

        ... preferentially in the order base name, 1st alias, 2nd alias, ... nth alias:
        >>> _get_value(f_c, Delta(c=3, ca=31, cb=32))
        (3, 'c')

        >>> _get_value(f_c, Delta(ca=31, cb=32))
        (31, 'ca')

        >>> _get_value(f_c, Delta(cb=32))
        ([32], 'cb')

        >>> print(_get_value(f_c, Delta()))
        (None, None)


    """

    key = f.name
    aliases = f.metadata.get("aliases", {})

    value, used_key = None, None

    if key in other.data.keys():
        value = other.data[key]
        used_key = key
    elif aliases:  # ... is not an empty dict
        for alias_key, wrapping_function in aliases.items():
            if alias_key in other.data:
                value = wrapping_function(other.data[alias_key])
                used_key = alias_key
                break  # we only evaluate the first match

    return value, used_key


def _get_field_names_and_aliases(s: State):
    """
    Get a list of field names and their aliases from a State object

    Args:
        s: a State object

    Returns: a list of field names and their aliases on `s`

    Examples:
        >>> from dataclasses import field
        >>> @dataclass(frozen=True)
        ... class SomeState(State):
        ...    l: List = field(default_factory=list)
        ...    m: List = field(default_factory=list)
        >>> _get_field_names_and_aliases(SomeState())
        ['l', 'm']

        >>> @dataclass(frozen=True)
        ... class SomeStateWithAliases(State):
        ...    l: List = field(default_factory=list, metadata={"aliases": {"l1": None, "l2": None}})
        ...    m: List = field(default_factory=list, metadata={"aliases": {"m1": None}})
        >>> _get_field_names_and_aliases(SomeStateWithAliases())
        ['l', 'l1', 'l2', 'm', 'm1']

    """
    result = []

    for f in fields(s):
        name = f.name
        result.append(name)

        aliases = f.metadata.get("aliases", {})
        result.extend(aliases)

    return result


class Delta(UserDict, Generic[S]):
    """
    Represents a delta where the base object determines the extension behavior.

    Examples:
        >>> from dataclasses import dataclass

        First we define the dataclass to act as the basis:
        >>> from typing import Optional, List
        >>> @dataclass(frozen=True)
        ... class ListState:
        ...     l: Optional[List] = None
        ...     m: Optional[List] = None
        ...
    """

    pass


Result = Delta
"""`Result` is an alias for `Delta`."""


@singledispatch
def extend(a, b):
    """
    Function to extend supported datatypes.

    """
    raise NotImplementedError("`extend` not implemented for %s, %s" % (a, b))


@extend.register(type(None))
def extend_none(a, b):
    """
    Examples:
        >>> extend(None, [])
        []

        >>> extend(None, [3])
        [3]
    """
    return b


@extend.register(list)
def extend_list(a, b):
    """
    Examples:
        >>> extend([], [])
        []

        >>> extend([1,2], [3])
        [1, 2, 3]
    """
    return a + b


@extend.register(pd.DataFrame)
def extend_pd_dataframe(a, b):
    """
    Examples:
        >>> extend(pd.DataFrame({"a": []}), pd.DataFrame({"a": []}))
        Empty DataFrame
        Columns: [a]
        Index: []

        >>> extend(pd.DataFrame({"a": [1,2,3]}), pd.DataFrame({"a": [4,5,6]}))
           a
        0  1
        1  2
        2  3
        3  4
        4  5
        5  6
    """
    return pd.concat((a, b), ignore_index=True)


@extend.register(np.ndarray)
def extend_np_ndarray(a, b):
    """
    Examples:
        >>> extend(np.array([(1,2,3), (4,5,6)]), np.array([(7,8,9)]))
        array([[1, 2, 3],
               [4, 5, 6],
               [7, 8, 9]])
    """
    return np.row_stack([a, b])


@extend.register(dict)
def extend_dict(a, b):
    """
    Examples:
        >>> extend({"a": "cats"}, {"b": "dogs"})
        {'a': 'cats', 'b': 'dogs'}
    """
    return dict(a, **b)


def append(a: List[T], b: T) -> List[T]:
    """
    Function to create a new list with an item appended to it.

    Examples:
        Given a starting list `a`:
        >>> a = [1, 2, 3]

        ... we can append a value:
        >>> append(a, 4)
        [1, 2, 3, 4]

        `a` is unchanged
        >>> a == [1, 2, 3]
        True

        Why not just use `list.append`? `list.append` mutates `a` in place, which we can't allow
        in the AER cycle – parts of the cycle rely on purely functional code which doesn't
        (accidentally or intentionally) manipulate existing data.
        >>> list.append(a, 4)  # not what we want
        >>> a
        [1, 2, 3, 4]
    """
    return a + [b]


def inputs_from_state(f):
    """Decorator to make target `f` into a function on a `State` and `**kwargs`.

    This wrapper makes it easier to pass arguments to a function from a State.

    It was inspired by the pytest "fixtures" mechanism.

    Args:
        f: the function which takes any arguments

    Returns: the function modified to take a State object as input and return a State object

    Examples:
        >>> from dataclasses import dataclass, field
        >>> import pandas as pd
        >>> from typing import List, Optional

        The `State` it operates on needs to have the metadata described in the state module:
        >>> @dataclass(frozen=True)
        ... class S(State):
        ...     conditions: List[int] = field(metadata={"delta": "replace"})

        We indicate the inputs required by the parameter names.
        The output must be a `Delta` object.
        >>> from autora.state.delta import Delta
        >>> @inputs_from_state
        ... def experimentalist(conditions):
        ...     new_conditions = [c + 10 for c in conditions]
        ...     return Delta(conditions=new_conditions)

        >>> experimentalist(S(conditions=[1,2,3,4]))
        S(conditions=[11, 12, 13, 14])

        >>> experimentalist(S(conditions=[101,102,103,104]))
        S(conditions=[111, 112, 113, 114])

        >>> from autora.variable import VariableCollection, Variable
        >>> from sklearn.base import BaseEstimator
        >>> from sklearn.linear_model import LinearRegression

        >>> @inputs_from_state
        ... def theorist(experiment_data: pd.DataFrame, variables: VariableCollection, **kwargs):
        ...     ivs = [v.name for v in variables.independent_variables]
        ...     dvs = [v.name for v in variables.dependent_variables]
        ...     X, y = experiment_data[ivs], experiment_data[dvs]
        ...     new_model = LinearRegression(fit_intercept=True).set_params(**kwargs).fit(X, y)
        ...     return Delta(model=new_model)

        >>> @dataclass(frozen=True)
        ... class T(State):
        ...     variables: VariableCollection  # field(metadata={"delta":... }) omitted ∴ immutable
        ...     experiment_data: pd.DataFrame = field(metadata={"delta": "extend"})
        ...     model: Optional[BaseEstimator] = field(metadata={"delta": "replace"}, default=None)

        >>> t = T(
        ...     variables=VariableCollection(independent_variables=[Variable("x")],
        ...                                  dependent_variables=[Variable("y")]),
        ...     experiment_data=pd.DataFrame({"x": [0,1,2,3,4], "y": [2,3,4,5,6]})
        ... )
        >>> t_prime = theorist(t)
        >>> t_prime.model.coef_, t_prime.model.intercept_
        (array([[1.]]), array([2.]))

        Arguments from the state can be overridden by passing them in as keyword arguments (kwargs):
        >>> theorist(t, experiment_data=pd.DataFrame({"x": [0,1,2,3], "y": [12,13,14,15]}))\\
        ...     .model.intercept_
        array([12.])

        ... and other arguments supported by the inner function can also be passed
        (if and only if the inner function allows for and handles `**kwargs` arguments alongside
        the values from the state).
        >>> theorist(t, fit_intercept=False).model.intercept_
        0.0

        Any parameters not provided by the state must be provided by default values or by the
        caller. If the default is specified:
        >>> @inputs_from_state
        ... def experimentalist(conditions, offset=25):
        ...     new_conditions = [c + offset for c in conditions]
        ...     return Delta(conditions=new_conditions)

        ... then it need not be passed.
        >>> experimentalist(S(conditions=[1,2,3,4]))
        S(conditions=[26, 27, 28, 29])

        If a default isn't specified:
        >>> @inputs_from_state
        ... def experimentalist(conditions, offset):
        ...     new_conditions = [c + offset for c in conditions]
        ...     return Delta(conditions=new_conditions)

        ... then calling the experimentalist without it will throw an error:
        >>> experimentalist(S(conditions=[1,2,3,4]))
        Traceback (most recent call last):
        ...
        TypeError: experimentalist() missing 1 required positional argument: 'offset'

        ... which can be fixed by passing the argument as a keyword to the wrapped function.
        >>> experimentalist(S(conditions=[1,2,3,4]), offset=2)
        S(conditions=[3, 4, 5, 6])

    """
    # Get the set of parameter names from function f's signature
    parameters_ = set(inspect.signature(f).parameters.keys())

    @wraps(f)
    def _f(state_: S, /, **kwargs) -> S:
        # Get the parameters needed which are available from the state_.
        # All others must be provided as kwargs or default values on f.
        assert dataclasses.is_dataclass(state_)
        from_state = parameters_.intersection(
            {i.name for i in dataclasses.fields(state_)}
        )
        arguments_from_state = {k: getattr(state_, k) for k in from_state}
        arguments = dict(arguments_from_state, **kwargs)
        delta = f(**arguments)
        new_state = state_ + delta
        return new_state

    return _f


def outputs_to_delta(*output: str):
    """
    Decorator factory to wrap outputs from a function as Deltas.

    Examples:
        >>> @outputs_to_delta("conditions")
        ... def add_five(x):
        ...     return [xi + 5 for xi in x]

        >>> add_five([1, 2, 3])
        {'conditions': [6, 7, 8]}

        >>> @outputs_to_delta("c")
        ... def add_six(conditions):
        ...     return [c + 5 for c in conditions]

        >>> add_six([1, 2, 3])
        {'c': [6, 7, 8]}

        >>> @outputs_to_delta("+1", "-1")
        ... def plus_minus_1(x):
        ...     a = [xi + 1 for xi in x]
        ...     b = [xi - 1 for xi in x]
        ...     return a, b

        >>> plus_minus_1([1, 2, 3])
        {'+1': [2, 3, 4], '-1': [0, 1, 2]}


        If the wrong number of values are specified for the return, then there might be errors.
        If multiple outputs are expected, but only a single output is returned, we get a warning:
        >>> @outputs_to_delta("1", "2")
        ... def returns_single_result_when_more_expected():
        ...     return "a"
        >>> returns_single_result_when_more_expected()  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
        Traceback (most recent call last):
        ...
        AssertionError: function `<function returns_single_result_when_more_expected at 0x...>`
        has to return multiple values to match `('1', '2')`. Got `a` instead.

        If multiple outputs are expected, but the wrong number are returned, we get a warning:
        >>> @outputs_to_delta("1", "2", "3")
        ... def returns_wrong_number_of_results():
        ...     return "a", "b"
        >>> returns_wrong_number_of_results()  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
        Traceback (most recent call last):
        ...
        AssertionError: function `<function returns_wrong_number_of_results at 0x...>`
        has to return exactly `3` values to match `('1', '2', '3')`. Got `('a', 'b')` instead.

        However, if a single output is expected, and multiple are returned, these are treated as
        a single object and no error occurs:
        >>> @outputs_to_delta("foo")
        ... def returns_a_tuple():
        ...     return "a", "b", "c"
        >>> returns_a_tuple()
        {'foo': ('a', 'b', 'c')}

        If we fail to specify output names, an error is returned immediately.
        >>> @outputs_to_delta()
        ... def decorator_missing_arguments():
        ...     return "a", "b", "c"
        Traceback (most recent call last):
        ...
        ValueError: `output` names must be specified.

    """

    def decorator(f):

        if len(output) == 0:
            raise ValueError("`output` names must be specified.")

        elif len(output) == 1:

            @wraps(f)
            def inner(*args, **kwargs):
                result = f(*args, **kwargs)
                delta = Delta(**{output[0]: result})
                return delta

        else:

            @wraps(f)
            def inner(*args, **kwargs):
                result = f(*args, **kwargs)
                assert isinstance(result, tuple), (
                    "function `%s` has to return multiple values "
                    "to match `%s`. Got `%s` instead." % (f, output, result)
                )
                assert len(output) == len(result), (
                    "function `%s` has to return "
                    "exactly `%s` values "
                    "to match `%s`. "
                    "Got `%s` instead."
                    "" % (f, len(output), output, result)
                )
                delta = Delta(**dict(zip(output, result)))
                return delta

        return inner

    return decorator


def on_state(
    function: Optional[Callable] = None, output: Optional[Sequence[str]] = None
):
    """Decorator (factory) to make target `function` into a function on a `State` and `**kwargs`.

    This combines the functionality of `outputs_to_delta` and `inputs_from_state`

    Args:
        function: the function to be wrapped
        output: list specifying State field names for the return values of `function`

    Returns:

    Examples:
        >>> from dataclasses import dataclass, field
        >>> import pandas as pd
        >>> from typing import List, Optional

        The `State` it operates on needs to have the metadata described in the state module:
        >>> @dataclass(frozen=True)
        ... class S(State):
        ...     conditions: List[int] = field(metadata={"delta": "replace"})

        We indicate the inputs required by the parameter names.
        >>> def add_ten(conditions):
        ...     return [c + 10 for c in conditions]
        >>> experimentalist = on_state(function=add_ten, output=["conditions"])

        >>> experimentalist(S(conditions=[1,2,3,4]))
        S(conditions=[11, 12, 13, 14])

        You can also wrap functions which return a Delta object natively, by omitting the `output`
        argument:
        >>> @on_state()
        ... def add_five(conditions):
        ...     return Delta(conditions=[c + 5 for c in conditions])

        >>> add_five(S(conditions=[1, 2, 3, 4]))
        S(conditions=[6, 7, 8, 9])

        You can also use the @on_state(output=[]) as a decorator:
        >>> @on_state(output=["conditions"])
        ... def add_six(conditions):
        ...     return [c + 6 for c in conditions]

        >>> add_six(S(conditions=[1, 2, 3, 4]))
        S(conditions=[7, 8, 9, 10])

    """

    def decorator(f):
        f_ = f
        if output is not None:
            f_ = outputs_to_delta(*output)(f_)
        f_ = inputs_from_state(f_)
        return f_

    if function is None:
        return decorator
    else:
        return decorator(function)
