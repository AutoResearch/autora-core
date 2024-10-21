# Workflow & State Mechanics

AutoRA includes core functionality for defining empirical research workflows. This core functionality is organized into these submodules:

- `autora.variable`, for representing experimental metadata describing the type and domain of variables
- `autora.state`, which underpins the unified `State` interface for interacting with experimentalists, experiment runners and 
  theorists
- `autora.serializer`, utilities for saving and loading `States`
- `autora.workflow`, command line tools for running experimentalists, experiment runners and theorists
- `autora.utils`, utilities and helper functions not linked to any specific core functionality  

It also provides some basic experimentalists in the `autora.experimentalist` submodule. However, most 
genuinely useful experimentalists and theorists are provided as optional dependencies to the `autora` package.

