"""Publication refuses altered artifacts and incomplete integration evidence."""
import copy
from pathlib import Path
import tempfile
import unittest

from .publish_release import sha256, verify_gate


class PublicationGateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.models = [{"experiment": f"model_{i}", "config_sha256": str(i)} for i in range(10)]
        reports = []
        for model in self.models:
            path = self.root / (model["experiment"] + "_seongeun.pkl")
            path.write_bytes(model["experiment"].encode())
            reports.append({**model, "sha256": sha256(path), "rows": 2833 * 72, "passed": True})
        package = self.root / "sunrun_inference.zip"
        package.write_bytes(b"verified package")
        self.receipt = {"passed": True, "models": reports, "package_sha256": sha256(package)}

    def test_complete_current_artifacts_pass(self):
        verify_gate(self.root, self.models, self.receipt)

    def test_changed_model_or_package_fails(self):
        for name in ("model_0_seongeun.pkl", "sunrun_inference.zip"):
            path = self.root / name
            previous = path.read_bytes()
            path.write_bytes(previous + b"changed")
            with self.assertRaises(ValueError):
                verify_gate(self.root, self.models, self.receipt)
            path.write_bytes(previous)

    def test_missing_duplicate_config_and_failed_checks_fail(self):
        invalid = []
        receipt = copy.deepcopy(self.receipt)
        receipt["models"].pop()
        invalid.append(receipt)
        receipt = copy.deepcopy(self.receipt)
        receipt["models"][1] = receipt["models"][0]
        invalid.append(receipt)
        receipt = copy.deepcopy(self.receipt)
        receipt["models"][0]["config_sha256"] = "wrong"
        invalid.append(receipt)
        receipt = copy.deepcopy(self.receipt)
        receipt["models"][0]["passed"] = False
        invalid.append(receipt)
        for receipt in invalid:
            with self.assertRaises(ValueError):
                verify_gate(self.root, self.models, receipt)


if __name__ == "__main__":
    unittest.main()
