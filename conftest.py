from hypothesis import HealthCheck, Verbosity, settings

settings.register_profile(
    "ci",
    verbosity=Verbosity.verbose,
    max_examples=1000,
    deadline=2000,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)
settings.register_profile("debug", max_examples=10, verbosity=Verbosity.verbose)
