#!/bin/bash
# Kette mit Modell (--kette) auf den Entwicklungssaetzen, je Modell eine Zeile.
# scripts/ liegt eine Ebene unter dem Projekt
cd "$(dirname "$0")/.."
for ckpt in "$@"; do
  for s in natural hand mkqa_dev mintaka_dev; do
    zeile=$(.venv/bin/python -m evaluation.run_chain --fragen evaluation/questions/$s.json --kette --fakten \
      --absaetze 3 --zeichen 1500 --strafe 1.05 --device mps --ckpt "$ckpt" 2>/dev/null | tail -1)
    echo "$(basename "$ckpt") $s: $zeile" | tee -a ergebnisse/lesen_auswertung.log
  done
done
