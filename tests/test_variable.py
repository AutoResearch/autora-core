from hypothesis import strategies as st

from autora.variable import ValueType, Variable, VariableCollection


@st.composite
def variable_strategy(draw):
    v = Variable(
        name=draw(st.text()),
        type=draw(st.sampled_from(ValueType)),
        value_range=draw(
            st.one_of(
                st.none(),
                st.tuples(st.integers(), st.integers()).filter(lambda v: v[0] <= v[1]),
                st.tuples(
                    st.floats(allow_nan=False), st.floats(allow_nan=False)
                ).filter(lambda v: v[0] <= v[1]),
            )
        ),
        allowed_values=draw(
            st.one_of(
                st.none(),
                st.lists(
                    st.one_of(st.booleans(), st.integers(), st.floats(allow_nan=False)),
                    unique=True,
                    min_size=1,
                ),
            )
        ),
        units=draw(st.text()),
        variable_label=draw(st.text()),
        rescale=draw(
            st.one_of(
                st.just(1),
                st.integers().filter(lambda v: v != 0),
                st.floats(allow_infinity=False, allow_nan=False).filter(
                    lambda v: v != 0
                ),
            )
        ),
        is_covariate=draw(st.just(False)),
    )
    return v


@st.composite
def variablecollection_strategy(draw):
    all_variables = draw(
        st.lists(variable_strategy(), unique_by=lambda v: v.name, min_size=0)
    )
    num_variables = len(all_variables)
    num_ivs = draw(st.integers(min_value=0, max_value=num_variables))
    # TODO: Add support for covariates
    num_ivs, num_dvs = num_ivs, num_variables - num_ivs
    vc = VariableCollection(
        independent_variables=all_variables[0:num_ivs],
        dependent_variables=all_variables[num_ivs : num_ivs + num_dvs],
        covariates=[],
    )
    return vc
