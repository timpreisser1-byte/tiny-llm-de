#!/bin/bash
# Kette mit Modell (--kette) auf den Entwicklungssaetzen, je Modell eine Zeile.
cd "$(dirname "$0")"
for ckpt in "$@"; do
  for s in echt hand mkqa_dev mintaka_dev; do
    zeile=$(.venv/bin/python chat_test.py --fragen chat_fragen_$s.json --kette --fakten \
      --absaetze 3 --zeichen 1500 --strafe 1.05 --device mps --ckpt "$ckpt" 2>/dev/null | tail -1)
    echo "$(basename "$ckpt") $s: $zeile" | tee -a ergebnisse/lesen_auswertung.log
  done
done
