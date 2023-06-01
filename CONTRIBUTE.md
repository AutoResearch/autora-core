# Contributor Guidelines

Suggested changes to the core should be submitted as follows, depending on their content:
- Fixes or new features closely associated with existing functionality or documentation of this package: a pull 
  request here
- New features which don't fit into the current module structure, or which are experimental and could lead to 
  instability for users: as a new namespace package, with a pull request to 
  [`autora`](https://github.com/autoresearch/autora) for inclusion as a default dependency.

!!! success
    Reach out to the core team about new core contributions to discuss how best to incorporate them by posting your 
    idea on the [discussions page](https://github.com/orgs/AutoResearch/discussions/categories/ideas).

Code contributions to this and other core packages should as a minimum:
- Have comprehensive documentation
- Have comprehensive test suites
- Follow standard python coding guidelines including PEP8
- Use the linters and checkers defined in the [.pre-commit-config.yaml](./.pre-commit-config.yaml)
- Run under all minor versions of python (e.g. 3.8, 3.9) allowed in 
  [`autora-core`](https://github.com/autoresearch/autora-core) and on all supported platforms (Linux, macOS and 
  Windows), with automated checks 
- Be compatible with all current AutoRA packages, i.e. cause no conflicts when installed alongside 
  `pip install --upgrade autora[all]`
