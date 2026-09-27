import json
from pathlib import Path
import tempfile
import unittest

import torch

from .model_manifest import manifest
from lm.model import Config, TinyLM
from lm.quantize import schreibe_datei
from lm.ternary import beschraenke


class ManifestTests(unittest.TestCase):
    def test_actual_export_size_and_complete_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = Config(vocab_size=32, n_layer=2, n_head=2, d_model=16,
                         d_ff=32, block_size=12, n_kv_head=1,
                         qk_norm=True, wert_residuum=True)
            model = TinyLM(cfg)
            beschraenke(model, "sherry")
            from dataclasses import asdict
            checkpoint = root / "model.pt"
            torch.save({"config": asdict(cfg), "model": model.state_dict(), "art": "sherry"}, checkpoint)
            before = checkpoint.read_bytes()
            report = manifest(checkpoint)
            names = {f"{name}.weight" for name, module in model.named_modules()
                     if type(module).__name__ == "SherryLinear"}
            reference = root / "reference.bin"
            schreibe_datei(str(reference), cfg, model.state_dict(), 4,
                          ternaer=names, packung="sherry")
            self.assertEqual(report["tlm1"]["temporary_export_bytes"], reference.stat().st_size)
            self.assertEqual(report["config"], asdict(cfg))
            self.assertIn("n_kv_head", report["tlm1"]["missing_config_fields"])
            self.assertTrue(report["tlm1"]["header_config_incomplete"])
            self.assertFalse(report["tlm1"]["deployment_ready"])
            self.assertEqual(report["memory_bytes"]["kv_cache_fp32"], 2 * 2 * 12 * 1 * 8 * 4)
            self.assertEqual(checkpoint.read_bytes(), before)
            json.dumps(report)

    def test_rejects_unsafe_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "unsafe.pt"
            torch.save({"unexpected": Config()}, checkpoint)
            with self.assertRaises(Exception) as raised:
                manifest(checkpoint)
            self.assertIn("Weights only load failed", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
