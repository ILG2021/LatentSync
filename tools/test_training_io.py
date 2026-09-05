"""CPU-only regression checks for checkpoint discovery and atomic publication."""
import ast
import os
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
import unittest


class CheckpointIOTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        source = Path(__file__).resolve().parents[1] / "scripts/train_unet.py"
        names = {"find_latest_checkpoint", "resolve_resume_ckpt_path", "save_checkpoint_atomic"}
        functions = [n for n in ast.parse(source.read_text(encoding="utf-8")).body
                     if isinstance(n, ast.FunctionDef) and n.name in names]
        self.scope = {"Path": Path, "re": re, "os": os}
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), self.scope)

    def checkpoint(self, relative, content=b"weights"):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def test_fixed_and_timestamped_steps(self):
        self.checkpoint("stage/checkpoints/checkpoint-500.pt")
        expected = self.checkpoint("stage/train-new/checkpoints/checkpoint-1000.pt")
        self.checkpoint("stage/checkpoints/checkpoint-2000.pt.tmp")
        self.checkpoint("stage/checkpoints/checkpoint-2000.training_state.pt")
        self.checkpoint("stage/checkpoints/checkpoint-3000.pt", b"")
        self.assertEqual(self.scope["find_latest_checkpoint"](self.root / "stage"), str(expected))

    def test_previous_stage_then_own_stage(self):
        class Config(dict):
            __getattr__ = dict.__getitem__
        config = Config(data=Config(train_output_dir=str(self.root / "stage2")),
                        ckpt=Config(resume_ckpt_path="auto", resume_search_dir=str(self.root / "stage1")))
        previous = self.checkpoint("stage1/checkpoints/checkpoint-9000.pt")
        resolve = self.scope["resolve_resume_ckpt_path"]
        self.assertEqual(resolve(config), (str(previous), False))
        own = self.checkpoint("stage2/checkpoints/checkpoint-100.pt")
        self.assertEqual(resolve(config), (str(own), True))
        config.ckpt["resume_ckpt_path"] = "explicit.pt"
        self.assertEqual(resolve(config), ("explicit.pt", True))

    def test_failed_save_preserves_existing_checkpoint(self):
        path = self.checkpoint("checkpoints/checkpoint-100.pt", b"old")
        def fail(state, temporary):
            Path(temporary).write_bytes(b"partial")
            raise OSError("disk full")
        self.scope["torch"] = SimpleNamespace(save=fail)
        with self.assertRaises(OSError):
            self.scope["save_checkpoint_atomic"]({}, path)
        self.assertEqual(path.read_bytes(), b"old")
        self.assertFalse(Path(str(path) + ".tmp").exists())

    def test_successful_save_publishes_checkpoint(self):
        path = self.checkpoint("checkpoints/checkpoint-100.pt", b"old")
        self.scope["torch"] = SimpleNamespace(save=lambda state, temporary: Path(temporary).write_bytes(b"new"))
        self.scope["save_checkpoint_atomic"]({}, path)
        self.assertEqual(path.read_bytes(), b"new")
        self.assertFalse(Path(str(path) + ".tmp").exists())


if __name__ == "__main__":
    unittest.main()
