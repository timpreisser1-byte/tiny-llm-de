"""Read-only checkpoint audit; measure the existing exporter in temporary storage.

Run from the project root:
  .venv/bin/python esp32_sd/model_manifest.py --checkpoint ckpt_8mb/sft_neu.pt --out report.json
This creates a JSON sidecar, never a deployment-ready firmware image.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from lm.model import Config, TinyLM
from lm.quantize import schreibe_datei
from lm.ternary import beschraenke

RESTRICTED = {"BitLinear", "SherryLinear", "SchwellLinear", "SherrySkala", "BinLinear", "BinSkala"}
HEADER_FIELDS = {"vocab_size", "n_layer", "n_head", "d_model", "d_ff", "block_size"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest(checkpoint, tokenizer=None, bits=4, psram_bytes=8 * 1024 * 1024):
    checkpoint = Path(checkpoint).resolve()
    if bits not in range(2, 9):
        raise ValueError("bits must be between 2 and 8")
    if psram_bytes <= 0:
        raise ValueError("psram_bytes must be positive")
    # Never fall back to unrestricted pickle deserialization.
    ck = torch.load(checkpoint, map_location="cpu", weights_only=True)
    cfg = Config(**ck["config"])
    model = TinyLM(cfg)
    art = ck.get("art") or ("ternaer" if ck.get("ternaer") else None)
    if art:
        beschraenke(model, art)
    model.load_state_dict(ck["model"], strict=True)
    names = {f"{name}.weight" for name, module in model.named_modules()
             if type(module).__name__ in RESTRICTED}
    packing = "sherry" if art == "sherry" else "trits"
    with tempfile.TemporaryDirectory(prefix="tiny-llm-manifest-") as tmp:
        exported = Path(tmp) / "measurement.bin"
        schreibe_datei(str(exported), cfg, ck["model"], bits,
                      ternaer=names, packung=packing)
        export_bytes = exported.stat().st_size
        export_hash = sha256(exported)
    config = asdict(cfg)
    kv_elements = 2 * cfg.n_layer * cfg.block_size * cfg.kv_heads * cfg.head_dim
    fp32_kv = kv_elements * 4
    subtotal = export_bytes + fp32_kv
    missing = sorted(set(config) - HEADER_FIELDS)
    result = {
        "schema": "tiny-llm-memory-manifest-v1",
        "checkpoint": {"path": str(checkpoint), "sha256": sha256(checkpoint)},
        "config": config,
        "quantization": {"training_kind": art, "matrix_packing": packing if names else None,
                         "other_matrix_bits": bits, "restricted_tensor_count": len(names)},
        "parameters_unique": model.num_params(),
        "tlm1": {"header_config_incomplete": bool(missing), "missing_config_fields": missing,
                 "temporary_export_bytes": export_bytes, "temporary_export_sha256": export_hash,
                 "measurement_method": "existing quantize.schreibe_datei, temporary file removed",
                 "export_inference_verified": False, "deployment_ready": False},
        "memory_bytes": {"psram_capacity_assumed": psram_bytes,
                         "packed_weights_and_headers": export_bytes,
                         "kv_cache_fp32": fp32_kv, "kv_cache_fp16": kv_elements * 2,
                         "weights_plus_fp32_kv": subtotal,
                         "remaining_before_other_allocations": psram_bytes - subtotal},
        "limitations": [
            "The subtotal assumes the packed export resides in PSRAM and KV values use FP32.",
            "Tokenizer runtime, activations, allocator overhead, stacks, SD search, decompression and audio are excluded.",
            "FP16 KV is an alternative calculation, not an implemented runtime feature.",
            "The sidecar supplies configuration but does not implement a C loader or inference engine.",
            "Export size does not establish answer quality or available runtime memory.",
        ],
    }
    if tokenizer is not None:
        tokenizer = Path(tokenizer).resolve()
        result["tokenizer"] = {"path": str(tokenizer), "sha256": sha256(tokenizer),
                               "file_bytes": tokenizer.stat().st_size,
                               "runtime_memory_bytes": None}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tokenizer", default=str(ROOT / "tokenizer/de_bpe.json"))
    parser.add_argument("--bits", type=int, default=4, choices=range(2, 9))
    args = parser.parse_args()
    report = manifest(args.checkpoint, args.tokenizer, args.bits)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Avoid overwriting a prior report or, accidentally, a checkpoint.
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(f"{output}: {report['tlm1']['temporary_export_bytes']:,} export bytes")


if __name__ == "__main__":
    main()
