#!/bin/bash
# Der komplette Spracherkenner auf dem eigenen Rechner - ohne Colab.
#
#   ./sprache_trainieren.sh              6 Shards (186 h), 20 Epochen
#   ./sprache_trainieren.sh 10 30        mehr Daten, mehr Epochen
#
# Laeuft unbeaufsichtigt durch. Abbruch mit Strg+C ist gefahrlos:
#   - fertige Shards werden beim Neustart uebersprungen
#   - das Training macht am letzten Checkpoint weiter (--fortsetzen)
#
# Warum das lokal Sinn ergibt: Die Merkmalsberechnung laeuft auf allen Kernen
# und ist auf einem M1 Pro rund viermal schneller als auf Colabs zwei Kernen.
# Nur beim Training ist die T4 vorn - dafuer laeuft der Mac ohne Zeitlimit
# und ohne Verbindungsabbruch.

set -e
SHARDS="${1:-6}"
EPOCHEN="${2:-20}"
# scripts/ liegt eine Ebene unter dem Projekt
cd "$(dirname "$0")/.."
PY=.venv/bin/python

echo "=== Spracherkennung: $SHARDS Shards (rund $((SHARDS * 31)) Stunden), $EPOCHEN Epochen ==="
echo

# ---------------------------------------------------------------- Daten
for ((i = 0; i < SHARDS; i++)); do
    if [ -f "data/asr/train$i.json" ]; then
        echo "Shard $((i + 1))/$SHARDS: schon fertig"
        continue
    fi
    echo "--- Shard $((i + 1))/$SHARDS"
    $PY -m speech.asr_data --portion "train$i"
    rm -f "data/asr/train$i.parquet"        # Rohdaten weg, Merkmale reichen
done

if [ ! -f data/asr/test.json ]; then
    echo "--- Testdaten"
    $PY -m speech.asr_data --portion test
    rm -f data/asr/test.parquet
fi

PORTIONEN=$(for ((i = 0; i < SHARDS; i++)); do printf "train%d," "$i"; done | sed 's/,$//')
echo
du -sh data/asr | sed 's/^/Merkmale: /'
echo

# ---------------------------------------------------------------- Training
echo "=== Training auf $PORTIONEN ==="
$PY -m speech.asr_model trainieren \
    --daten "$PORTIONEN" \
    --out ckpt_asr/ctc_gross.pt \
    --epochen "$EPOCHEN" --batch 16 --kanaele 256 --bloecke 8 \
    --zeige 2 --sichern 1 --fortsetzen

# ---------------------------------------------------------------- Bewertung
echo
echo "=== Wortfehlerrate auf ungesehenen Testdaten ==="
$PY -m speech.asr_model testen --daten test --modell ckpt_asr/ctc_gross.pt --anzahl 300

echo
echo "Fertig. Modell: ckpt_asr/ctc_gross.pt"
echo "Eigene Aufnahme testen:  $PY -m speech.asr_model testen --datei meine.wav --modell ckpt_asr/ctc_gross.pt"
