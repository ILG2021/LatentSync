"""CPU-only regression tests for fail-closed training-state restoration.

Load the actual helper via AST to avoid importing GPU/video training dependencies.
Run with: python -m unittest discover -s tools -p test_training_state_restore.py
"""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock


class TrainingStateRestoreTests(unittest.TestCase):
    def setUp(self):
        source = Path(__file__).resolve().parents[1] / "scripts" / "train_unet.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        helper = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "load_training_state")
        self.optimizer = Mock()
        self.scheduler = Mock()
        self.scaler = Mock()
        self.state = {
            "optimizer_type": f"{type(self.optimizer).__module__}.{type(self.optimizer).__qualname__}",
            "optimizer": {"sentinel": "moments"},
            "lr_scheduler": {"sentinel": "schedule"},
            "scaler": {"sentinel": "scale"},
        }
        self.exists = Mock(return_value=True)
        namespace = {
            "os": SimpleNamespace(path=SimpleNamespace(isfile=self.exists)),
            "torch": SimpleNamespace(load=Mock(return_value=self.state)),
            "training_state_path": lambda path: path + ".training_state.pt",
        }
        exec(compile(ast.Module(body=[helper], type_ignores=[]), str(source), "exec"), namespace)
        self.restore = namespace["load_training_state"]

    def test_missing_sidecar_leaves_fresh_objects(self):
        self.exists.return_value = False
        self.assertFalse(self.restore("checkpoint", self.optimizer, self.scheduler, self.scaler))
        self.optimizer.load_state_dict.assert_not_called()

    def test_matching_state_restores_all_components(self):
        self.assertTrue(self.restore("checkpoint", self.optimizer, self.scheduler, self.scaler))
        self.optimizer.load_state_dict.assert_called_once_with(self.state["optimizer"])
        self.scheduler.load_state_dict.assert_called_once_with(self.state["lr_scheduler"])
        self.scaler.load_state_dict.assert_called_once_with(self.state["scaler"])

    def test_legacy_and_wrong_optimizer_rejected_before_mutation(self):
        for optimizer_type in (None, "bitsandbytes.optim.adamw.AdamW8bit"):
            with self.subTest(optimizer_type=optimizer_type):
                self.state["optimizer_type"] = optimizer_type
                with self.assertRaisesRegex(RuntimeError, "Optimizer type"):
                    self.restore("checkpoint", self.optimizer, self.scheduler, self.scaler)
                self.optimizer.load_state_dict.assert_not_called()

    def test_partial_restore_failure_is_fatal(self):
        self.scheduler.load_state_dict.side_effect = ValueError("invalid scheduler")
        with self.assertRaisesRegex(RuntimeError, "invalid scheduler"):
            self.restore("checkpoint", self.optimizer, self.scheduler, self.scaler)
        self.optimizer.load_state_dict.assert_called_once()
        self.scaler.load_state_dict.assert_not_called()


if __name__ == "__main__":
    unittest.main()
