"""Reject rehashed family/diagnostic mutations without using validation targets."""

from copy import deepcopy
import unittest

from mlsys_sim.cost_calibration import ALGORITHM, cost_data_sha256, validate_analytical_costs
from mlsys_sim.schema import SimulationError
from test_step_cost_calibration import fixture, fitted


def rehash(artifact):
    artifact["fit_sha256"] = cost_data_sha256(
        {key: value for key, value in artifact.items() if key != "fit_sha256"})


def replace_cost(artifact, **values):
    # Keep the executable patch internally consistent, rather than relying on
    # object aliasing to trigger a generic patch-check failure.
    artifact["cost_specs"][0].update(values)
    artifact["calibration_patch"]["analytical_costs"] = deepcopy(artifact["cost_specs"])


class StepFitIntegrityTests(unittest.TestCase):
    def test_old_algorithm_cannot_launder_whole_step_metadata(self):
        data = fixture()
        artifact = fitted(data)
        artifact["algorithm"] = ALGORITHM
        replace_cost(artifact, attention_scale=2.1)
        rehash(artifact)
        with self.assertRaises(SimulationError):
            validate_analytical_costs(data, artifact)

    def test_stripping_step_metadata_cannot_invent_component_training(self):
        data = fixture()
        artifact = fitted(data)
        artifact["algorithm"] = ALGORITHM
        artifact.pop("step_fit_spec")
        artifact.pop("fit_spec_sha256")
        diagnostic = artifact["diagnostics"][0]
        keep = {"domain_sha256", "domain", "phase", "model_id", "training_record_ids"}
        artifact["diagnostics"] = [{key: value for key, value in diagnostic.items() if key in keep}]
        replace_cost(artifact, attention_scale=2.1)
        rehash(artifact)
        self.assertTrue(all(row["kind"] == "step_wall" for row in data["records"]))
        with self.assertRaises(SimulationError):
            validate_analytical_costs(data, artifact)

    def test_shared_scale_must_match_frozen_affine_diagnostic(self):
        data = fixture()
        artifact = fitted(data)
        replace_cost(artifact, nonattention_scale=2.1, attention_scale=2.1)
        rehash(artifact)
        with self.assertRaises(SimulationError):
            validate_analytical_costs(data, artifact)

    def test_intercept_must_match_frozen_affine_diagnostic(self):
        data = fixture()
        artifact = fitted(data)
        replace_cost(artifact, runtime_fixed_us=4100)
        rehash(artifact)
        with self.assertRaises(SimulationError):
            validate_analytical_costs(data, artifact)

    def test_solver_and_cell_weighting_claims_are_checked(self):
        data = fixture()
        for field, value in (("solver", "unconstrained"), ("equal_weight_cell_count", 400)):
            with self.subTest(field=field):
                artifact = fitted(data)
                artifact["diagnostics"][0]["whole_step_affine"][field] = value
                rehash(artifact)
                with self.assertRaises(SimulationError):
                    validate_analytical_costs(data, artifact)

    def test_retained_training_errors_match_the_actual_costs(self):
        data = fixture()
        artifact = fitted(data)
        artifact["diagnostics"][0]["training_wall_errors"]["record_mape"] = 0.5
        rehash(artifact)
        with self.assertRaises(SimulationError):
            validate_analytical_costs(data, artifact)


if __name__ == "__main__":
    unittest.main()
