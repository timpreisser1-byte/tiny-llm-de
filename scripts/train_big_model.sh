#!/bin/sh
# Dasselbe Geraetemodell, aber satt UND von Anfang an mit 1,25 Bit:
# gemessen kostet Sherry nachtraeglich zwoelf Punkte Lesefaehigkeit,
# als Trainingsart hoechstens zwei.
# Satt: 704 Mio. Token statt 51, davon 22 %
# Lesestoff in achtzehn Formen. Wartet auf den Erkenner - beide zugleich
# auf MPS ist noch nie gutgegangen.
# scripts/ liegt eine Ebene unter dem Projekt
cd "$(dirname "$0")/.."
P=.venv/bin/python
until grep -q "KETTE FINAL FERTIG" kette_final.log 2>/dev/null; do sleep 120; done

echo "=== 1) 13M satt trainiert  $(date +%H:%M) ==="
$P -m training.pretrain --preset 15m --sherry --qk-norm --wert-residuum --kv-head 1 \
    --kontext 384 --steps 17000 --batch-size 24 --lr 4e-3 --plan wsd \
    --warmup 400 --eval-every 1000 --window-tokens 80000000 --window-steps 400 \
    --resume --data-dir data_8mb --ckpt-dir ckpt_8mb_gross

echo ""
echo "=== 2) Feintuning, kurze Antworten  $(date +%H:%M) ==="
$P -m training.sft --init ckpt_8mb_gross/best.pt --out ckpt_8mb_gross/sft.pt \
    --epochs 2 --batch-size 16 --lr 1e-4 \
    --data data/sft/hochwertig_gekappt.jsonl data/sft/mehrfakten.jsonl:3 \
           data/sft/smalltalk_de.jsonl:40 data/sft/grenzen_de.jsonl:25 \
           data/sft/seed_de.jsonl:40 data/sft/elektronik_de.jsonl:30 \
           data/sft/ratgeber_de.jsonl:30 data/sft/kurzkontext.jsonl:4

echo ""
echo "=== 3) Lesetest, fremde Form  $(date +%H:%M) ==="
$P -m evaluation.reading_test --device cpu --n 300 --max-zeichen 450 --form fremd \
    --ckpt ckpt_8mb/sft_kurz.pt --ckpt ckpt_8mb_gross/sft.pt

echo ""
echo "=== 4) Das Gerät  $(date +%H:%M) ==="
$P -m lm.inference --fragen --ckpt ckpt_8mb_gross/sft.pt

echo ""
echo "=== 5) Prüfungen  $(date +%H:%M) ==="
$P -m evaluation.battery --ckpt ckpt_8mb_gross/sft.pt
$P -m evaluation.broad100 --ckpt ckpt_8mb_gross/sft.pt --device cpu

echo ""
echo "=== KETTE GROSS FERTIG  $(date +%H:%M) ==="
