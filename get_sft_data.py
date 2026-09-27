"""Schritt 5a: deutsche Frage-Antwort-Daten fuer das Chat-Finetuning.

Laedt frei nutzbare deutsche Instruktions- und Dialogdatensaetze von Hugging
Face und schreibt sie einheitlich nach data/sft/train.jsonl:

    {"messages": [{"role": "user", "content": "..."},
                  {"role": "assistant", "content": "..."}]}

Gelesen und gefiltert wird in dialogdaten.py - dieselbe Stelle, die auch
build_mix.py fuer das Pretraining benutzt.

    python get_sft_data.py
"""

import argparse
import json
import os
import random
import urllib.request

import dialogdaten

BASIS = "https://huggingface.co/datasets"
# Name -> (Dateiendung, URL)
QUELLEN = {
    # echte Gespraeche
    "german-conv": ("parquet", f"{BASIS}/jonas-is-coding/german-conversations/resolve/main/data/train-00000-of-00001.parquet"),
    "oasst-chains_de": ("parquet", f"{BASIS}/m-ric/Open_Assistant_Chains_German_Translation/resolve/main/data/german-00000-of-00001.parquet"),
    "sharegpt_de": ("json", f"{BASIS}/FreedomIntelligence/sharegpt-deutsch/resolve/main/sharegpt-deutsch.json"),
    # muttersprachlich deutsch, von Menschen geschrieben und bewertet
    "oasst2_trees": ("jsonl.gz", f"{BASIS}/OpenAssistant/oasst2/resolve/main/2023-11-05_oasst2_ready.trees.jsonl.gz"),
    "oasst1-qa_de": ("parquet", f"{BASIS}/AgentWaller/german-oasst1-qa-format/resolve/main/data/train-00000-of-00001-58a88ad0a3b65c01.parquet"),
    # Instruktionsdaten
    "ultra-chat_de": ("json", f"{BASIS}/mayflowergmbh/ultra-chat_de/resolve/main/ultra_chat_german.json"),
    "dolphin_de": ("json", f"{BASIS}/mayflowergmbh/dolphin_de/resolve/main/dolphin.json"),
    "evol-instruct_de": ("json", f"{BASIS}/mayflowergmbh/evol-instruct_de/resolve/main/evol_instruct_de.json"),
    "airoboros-3.0_de": ("json", f"{BASIS}/mayflowergmbh/airoboros-3.0_de/resolve/main/airoboros_3.json"),
    "openschnabeltier_de": ("json", f"{BASIS}/mayflowergmbh/openschnabeltier_de/resolve/main/openschnabeltier.json"),
    "dolly_de": ("json", f"{BASIS}/mayflowergmbh/dolly-15k_de/resolve/main/dolly_de.json"),
    "oasst_de": ("json", f"{BASIS}/mayflowergmbh/oasst_de/resolve/main/oasst_de.json"),
    "alpaca_de": ("json", f"{BASIS}/FreedomIntelligence/alpaca-gpt4-deutsch/resolve/main/alpaca-gpt4-deutsch.json"),
}
# Gespraechsnahe Daten und eigene Beispiele staerker gewichten
GEWICHTE = {"oasst2_trees": 6, "german-conv": 3, "oasst-chains_de": 3,
            "sharegpt_de": 2, "dolly_de": 2, "oasst_de": 2,
            "seed_de": 30, "ratgeber_de": 30, "kontext_qa": 3}
# Lokale Zusatzdateien (nicht von Hugging Face): eigene Beispiele und die
# Kontextfragen aus wissen_daten.py
LOKAL = ["seed_de.jsonl", "ratgeber_de.jsonl", "kontext_qa.jsonl"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/sft")
    ap.add_argument("--out", default="data/sft/train.jsonl")
    ap.add_argument("--seed-file", default="data/sft/seed_de.jsonl")
    args = ap.parse_args()

    os.makedirs(args.dir, exist_ok=True)
    alle = []

    for name, (endung, url) in QUELLEN.items():
        pfad = os.path.join(args.dir, f"{name}.{endung}")
        if not os.path.exists(pfad) or os.path.getsize(pfad) == 0:
            print(f"  lade {name} ...", flush=True)
            urllib.request.urlretrieve(url, pfad)
        g = GEWICHTE.get(name, 1)
        n = 0
        for msgs in dialogdaten.lade(pfad):
            alle.extend([msgs] * g)
            n += g
        print(f"  {name:22} {n:>9,} Dialoge", flush=True)

    for name in LOKAL:
        pfad = os.path.join(args.dir, name)
        if not os.path.exists(pfad):
            continue
        kurz = name.split(".")[0]
        eigene = list(dialogdaten.lade(pfad))
        g = GEWICHTE.get(kurz, 1)
        alle.extend(eigene * g)
        print(f"  {kurz:22} {len(eigene)*g:>9,} Dialoge ({len(eigene)} x {g})")

    random.seed(0)
    random.shuffle(alle)
    with open(args.out, "w", encoding="utf-8") as f:
        for msgs in alle:
            f.write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")

    mehrfach = sum(1 for m in alle if len(m) > 2)
    print(f"\n{len(alle):,} Dialoge -> {args.out}")
    print(f"  davon mehrstufig (mit Verlauf): {mehrfach:,}")


if __name__ == "__main__":
    main()
