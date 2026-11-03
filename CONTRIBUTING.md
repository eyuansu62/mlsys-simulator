# Contributing

The repository structure is being prepared for its initial release. License
selection is pending; settle it before public release and accepting external
contributions.

## Set up a checkout

Use Python 3.11 or newer and a virtual environment:

```bash
python -m venv .venv
# Activate the environment using the command for your shell.
python -m pip install -e .
python -m pip install -r requirements-dev.txt
```

## Make a change

Start from a small reproducible workload and identify whether the change affects
configuration, performance estimates, memory accounting, scheduling or metrics.
Keep those boundaries explicit. Add a regression case before fixing a behavior
bug, and preserve deterministic results for unchanged inputs.

For a new mechanism, update its support check and the capability matrix. Unknown
or unsupported options must fail clearly. A performance-model change must state
its assumptions and units; synthetic fixtures do not establish hardware
accuracy. Online scheduling and victim selection must not inspect future output
lengths from the replay oracle.

Run the checks relevant to the change, then the full suite before a merge:

```bash
python -m unittest discover -s tests -v
ruff check .
mypy --check-untyped-defs --ignore-missing-imports mlsys_sim
python -m build
twine check dist/*
```

Keep runtime dependencies empty unless a concrete requirement justifies a new
one. Build and development tools belong in their separate configuration.

## Report results

An issue should include the command, a minimal synthetic configuration, expected
behavior, observed behavior and Python version. A pull request should explain
the trigger and resulting behavior, list validation performed, and describe any
remaining modeling limits. Use the bug or feature template when available.

Commit messages follow the repository's Lore format: explain why the change was
made, then add useful native trailers such as `Constraint:`, `Rejected:`,
`Tested:` and `Not-tested:`. Example:

```text
Keep timed-out requests from releasing in-flight KV blocks

The completion boundary still owns the reserved blocks after a timeout.
Release them only after that boundary has been handled.

Confidence: high
Scope-risk: narrow
Tested: In-flight timeout and deterministic replay regression tests
Not-tested: Hardware runtime integration
```

Release preparation is documented in [docs/releasing.md](docs/releasing.md).
