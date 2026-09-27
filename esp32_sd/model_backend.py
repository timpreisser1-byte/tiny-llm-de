"""Optional desktop model backend for retrieval-augmented answers.

This is a host-side reference, not an ESP32 runtime.  It loads an existing
PyTorch checkpoint on CPU and keeps the prompt inside the model context window
while preserving the user's question.
"""

from pathlib import Path
import sys

import torch
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chat_format import build_prompt, mit_wissen
from model import Config, TinyLM
from ternaer import beschraenke


class ModelBackend:
    def __init__(self, checkpoint, tokenizer, max_new_tokens=60):
        torch.set_num_threads(min(2, max(1, torch.get_num_threads())))
        self.device = torch.device("cpu")
        self.max_new_tokens = max_new_tokens
        self.tokenizer = Tokenizer.from_file(str(tokenizer))
        ck = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.cfg = Config(**ck["config"])
        self.model = TinyLM(self.cfg)
        art = ck.get("art") or ("ternaer" if ck.get("ternaer") else None)
        if art:
            beschraenke(self.model, art)
        self.model.load_state_dict(ck["model"], strict=True)
        self.model.to(self.device).eval()
        self.eos = self.tokenizer.token_to_id("</s>")

    def _encode_prompt(self, question, context):
        room = max(16, self.cfg.block_size - self.max_new_tokens - 4)

        def encode_with(short_context):
            text = mit_wissen(question, short_context) if short_context else question
            return self.tokenizer.encode(
                build_prompt([{"role": "user", "content": text}]),
                add_special_tokens=False,
            ).ids

        ids_without_context = encode_with("")
        if len(ids_without_context) > room:
            raise ValueError("Die Frage allein ist zu lang fuer das Modellfenster.")
        if not context:
            return ids_without_context
        if len(encode_with(context)) <= room:
            return encode_with(context)

        low, high = 0, len(context)
        best = ""
        while low <= high:
            mid = (low + high) // 2
            trial = context[:mid]
            if len(encode_with(trial)) <= room:
                best = trial
                low = mid + 1
            else:
                high = mid - 1
        return encode_with(best)

    @torch.no_grad()
    def generate(self, question, context):
        ids = self._encode_prompt(question, context)
        x = torch.tensor([ids], device=self.device)
        output = [
            token
            for token in self.model.generate(
                x,
                max_new_tokens=self.max_new_tokens,
                temperature=0.0,
                top_k=20,
                repetition_penalty=1.15,
                eos_id=self.eos,
            )
            if token != self.eos
        ]
        return self.tokenizer.decode(output).strip()
