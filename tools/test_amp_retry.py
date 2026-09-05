"""Exercise the actual AMP update branch with CPU-only test doubles."""
import ast
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock


class AmpRetryTests(unittest.TestCase):
    def run_branch(self, failures):
        source = Path(__file__).resolve().parents[1] / "scripts/train_unet.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        loop = next(n for n in ast.walk(tree) if isinstance(n, ast.For)
                    and isinstance(n.target, ast.Name) and n.target.id == "amp_attempt")
        branch = next(n for n in loop.body if isinstance(n, ast.If)
                      and ast.unparse(n.test) == "config.run.mixed_precision_training")
        # Keep the real update/overflow branch and loop bound; substitute only the forward graph.
        loop.body = [branch, ast.Break()]
        module = ast.fix_missing_locations(ast.Module(body=[loop], type_ignores=[]))
        parameter = NS(grad=None)
        self.optimizer = Mock()
        self.clipper = Mock(return_value=1.0)
        self.scaler = Mock()
        self.attempts = 0
        scale = [65536.0]

        def backward():
            self.attempts += 1
            parameter.grad = self.attempts > failures

        def update():
            if not parameter.grad:
                scale[0] /= 2

        self.scaler.scale.return_value.backward.side_effect = backward
        self.scaler.get_scale.side_effect = lambda: scale[0]
        self.scaler.update.side_effect = update
        self.scaler.step.side_effect = lambda optimizer: optimizer.step()
        scope = {
            "config": NS(run=NS(mixed_precision_training=True), optimizer=NS(max_grad_norm=1.0)),
            "scaler": self.scaler, "optimizer": self.optimizer, "loss": object(),
            "trainable_named_params": [("weight", parameter)], "trainable_params": [parameter],
            "global_step": 1000, "logger": Mock(),
            "torch": NS(isfinite=lambda value: NS(all=lambda: NS(item=lambda: value)),
                        nn=NS(utils=NS(clip_grad_norm_=self.clipper))),
        }
        exec(compile(module, str(source), "exec"), scope)

    def test_overflow_retries_without_optimizer_update(self):
        self.run_branch(failures=2)
        self.assertEqual(self.attempts, 3)
        self.optimizer.step.assert_called_once()
        self.clipper.assert_called_once()
        self.assertEqual(self.optimizer.zero_grad.call_count, 2)

    def test_persistent_overflow_stops_without_update(self):
        with self.assertRaisesRegex(RuntimeError, "17 attempts"):
            self.run_branch(failures=100)
        self.assertEqual(self.attempts, 17)
        self.optimizer.step.assert_not_called()
        self.clipper.assert_not_called()


if __name__ == "__main__":
    unittest.main()
