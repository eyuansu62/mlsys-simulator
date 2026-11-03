# Release preparation

This checkout has a private GitHub repository and V1 software artifacts. License
selection and a PyPI release remain pending. The workflow tests and builds
artifacts; it does not publish them.

## Resolve release identity

- Select a license with the owner, add its actual text, and update package
  metadata and both READMEs consistently.
- Confirm the distribution name `mlsys-simulator` is available and appropriate
  before a PyPI upload. The Python import name is `mlsys_sim`.
- Confirm public repository visibility only after the license decision, then
  update repository and release metadata consistently.
- Confirm the intended version in `mlsys_sim.__version__` and update the
  changelog. A local version number is not a published release.

## Verify the candidate

From a clean checkout, run:

```bash
python -m pip install -e .
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
ruff check .
mypy --check-untyped-defs --ignore-missing-imports mlsys_sim
python -m build
twine check dist/*
python scripts/verify_distribution.py
```

Build in a fresh output directory so old distributions are not included in a
release by accident. Inspect the wheel and source distribution: the wheel should
contain the library and console entry point; the source distribution should
also retain tests, examples and public documentation. Exclude generated runs,
local machine paths, caches and task notes.

Install the wheel in a separate environment outside the source checkout. Verify
`import mlsys_sim`, `mlsys-sim --help`, a synthetic simulation and an analytical
prediction. Confirm the installed version and metadata agree. Also build a
wheel from the source distribution to check that it contains the required build
inputs.

The CI matrix checks supported Python versions when it runs remotely. Local
checks alone do not establish that every matrix environment passed. Record the
actual results and any untested platform in the validation report.

## Publish after the candidate is approved

After identity, license and validation are complete, configure the selected
remote and push the reviewed source. Run CI and review its results before
creating the release tag. TestPyPI or PyPI upload is a separate explicit action;
this repository contains no automatic publication job.

A release must describe the current model scope and calibration status. Keep
synthetic correctness checks separate from measured hardware validation, and
do not describe PP analysis as online PP simulation.
