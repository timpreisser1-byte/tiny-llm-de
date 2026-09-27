"""Das ganze Gerät in einer Datei: Frage rein, Antwort raus.

Hier laufen die Teile zum ersten Mal zusammen, so wie sie spaeter auf dem
ESP32-P4 zusammenlaufen sollen:

    Frage  ->  Wortindex auf der Karte  ->  Absatz  ->  Sprachmodell  ->  Antwort

Das Sprachmodell weiss nichts. Es liest. Deshalb steht und faellt die
Antwort damit, ob die Suche den richtigen Absatz findet - und genau das
laesst sich hier zum ersten Mal getrennt messen:

    Suche gut, Antwort gut       das Verfahren traegt
    Suche gut, Antwort schlecht  das Modell liest nicht
    Suche schlecht               der Index ist das Problem, nicht das Modell

    python geraet.py "Wie hoch ist der Eiffelturm?"
    python geraet.py --fragen           # feste Liste durchlaufen
"""

import argparse
import sys
import time

import torch

from chat_format import build_prompt, frage_zuerst, mit_wissen
from model import Config, TinyLM
from train import pick_device
from wissen_index import Wissensbasis

GRUEN, GELB, GRAU, FETT, AUS = "\033[92m", "\033[93m", "\033[90m", "\033[1m", "\033[0m"

FRAGEN = [
    "Wo liegt Quedlinburg?",
    "Wie hoch ist der Eiffelturm?",
    "Wann wurde die Berliner Mauer gebaut?",
    "Wer hat die Relativitätstheorie entwickelt?",
    "Wie viele Einwohner hat Hamburg?",
    "Was ist ein Transistor?",
    "Wann wurde Goethe geboren?",
    "Wie lang ist der Rhein?",
    "Was ist die Hauptstadt von Bayern?",
    "Wie tief ist der Bodensee?",
]


def lade(pfad, device):
    ck = torch.load(pfad, map_location="cpu", weights_only=False)
    cfg = Config(**ck["config"])
    netz = TinyLM(cfg)
    art = ck.get("art") or ("ternaer" if ck.get("ternaer") else None)
    if art:
        from ternaer import beschraenke
        beschraenke(netz, art)
    netz.load_state_dict(ck["model"])
    return netz.to(device).eval(), cfg


def antworte(netz, tok, cfg, device, frage, wissen, laenge=60, strafe=1.15,
             reihenfolge="wissen"):
    bau = frage_zuerst if reihenfolge == "frage" else mit_wissen
    text = bau(frage, wissen) if wissen else frage
    # Platz fuer die Antwort freihalten. Wird bis genau block_size gekuerzt,
    # bleibt kein Fenster zum Erzeugen uebrig und RoPE laeuft ins Leere.
    raum = max(16, cfg.block_size - laenge - 4)
    ids = tok.encode(build_prompt([{"role": "user", "content": text}]),
                     add_special_tokens=False).ids[-raum:]
    x = torch.tensor([ids], device=device)
    eos = tok.token_to_id("</s>")
    # Achtung Wiederholungsstrafe: Bei einer Aufgabe, deren richtige Antwort
    # darin besteht, Woerter aus dem Text zu wiederholen, arbeitet sie gegen
    # das Ziel. Sie drueckt bereits benutzte Wortanfaenge nieder und erzeugt
    # Bruchstuecke wie "Deutschlandsster" aus "Deutschlands hoechster".
    raus = [t for t in netz.generate(x, max_new_tokens=laenge, temperature=0.0,
                                     top_k=20, repetition_penalty=strafe,
                                     eos_id=eos) if t != eos]
    return tok.decode(raus).strip()


@torch.no_grad()
def antworte_kopie(netz, tok, cfg, device, frage, wissen, laenge=24,
                   min_laenge=4):
    """Die Antwort darf nur ein woertliches Stueck aus dem Absatz sein.

    Warum ueberhaupt: Zwei von drei Antworten des freien Modells enthalten
    einen Namen, der im Absatz nirgends steht - "Maerce von Bismarck wurde
    1749 in Frankfurt geboren". Die Literatur beziffert das allgemein:
    extraktive Systeme erfinden in 3 bis 8 Prozent der Faelle, frei
    erzeugende in 15 bis 25. Wer kopiert, kann nichts erfinden.

    Das Verfahren ist eine Einschraenkung beim Dekodieren, kein neues
    Training. Bei jedem Schritt wird mitgefuehrt, an welchen Stellen des
    Absatzes das bisher Geschriebene woertlich vorkommt; erlaubt sind nur
    die Tokens, die dort weitergehen. Das ist derselbe Gedanke wie der
    Woerterbuchbaum in `asr_strahl.py`, nur ueber den Absatz statt ueber
    ein Lexikon - und kostet nichts ausser einer Mengenoperation je Schritt.

    Das Modell darf jederzeit aufhoeren: EOS steht immer zur Wahl.
    """
    text = mit_wissen(frage, wissen) if wissen else frage
    raum = max(16, cfg.block_size - laenge - 4)
    ids = tok.encode(build_prompt([{"role": "user", "content": text}]),
                     add_special_tokens=False).ids[-raum:]
    quelle = tok.encode(wissen, add_special_tokens=False).ids
    if not quelle:
        return ""
    eos = tok.token_to_id("</s>")

    x = torch.tensor([ids], device=device)
    stellen = set(range(len(quelle)))     # ueberall darf eine Spanne beginnen
    raus, kv = [], None
    for _ in range(laenge):
        logits, kv = netz(x, kv_caches=kv)
        werte = logits[0, -1]
        erlaubt = sorted({quelle[p] for p in stellen})
        if not erlaubt:
            break
        # EOS steht daneben, aber nicht sofort: Sonst schreibt das Modell ein
        # einziges Token und hoert auf ("E", "Rhein", "1."). Gemessen: ohne
        # Mindestlaenge sinkt die Antwortquote von 33 auf 3 Prozent.
        kandidaten = erlaubt + ([eos] if len(raus) >= min_laenge else [])
        bester = kandidaten[int(werte[torch.tensor(kandidaten,
                                                   device=device)].argmax())]
        if bester == eos:
            break
        raus.append(bester)
        stellen = {p + 1 for p in stellen
                   if quelle[p] == bester and p + 1 < len(quelle)}
        if not stellen:
            break
        x = torch.tensor([[bester]], device=device)
    return tok.decode(raus).strip()


@torch.no_grad()
def antworte_spanne(netz, tok, cfg, device, frage, wissen, max_len=8, alpha=0.7):
    """Alle Spannen des Absatzes bewerten und die beste als Antwort nehmen.

    Gieriges Kopieren scheitert daran, dass das Modell auf ganze Saetze
    trainiert ist: Gefragt nach der Hoehe des Eiffelturms faengt es an, den
    Absatz von vorn abzuschreiben - "Der Eiffelt..." - statt zu "330 Meter"
    zu springen. Der erste Schritt entscheidet alles, und er faellt falsch.

    Hier faellt keine Entscheidung Token fuer Token. Jede zusammenhaengende
    Spanne bekommt eine Note, und die beste gewinnt:

        Note = ( log P(Spanne) + log P(Ende danach) ) / (L+1)^alpha

    Der zweite Summand ist der Kniff. Er fragt nicht nur "passt diese Spanne
    als Fortsetzung", sondern "**hoert die Antwort hier auf**". Ohne ihn
    gewinnen Satzanfaenge, weil danach immer plausibel etwas weitergeht.

    Der Nenner ist die Laengenausgleichung aus der Strahlsuche: ohne sie
    gewinnt stets die kuerzeste Spanne, mit alpha=1 die laengste.

    Gerechnet wird alles in einem Durchgang: Von jeder Startstelle laeuft
    dieselbe Schrittfolge, und nach jedem Schritt steht die Note fuer diese
    Laenge fest. Das kostet max_len Durchgaenge ueber je ein Token - nicht
    einen je Spanne.
    """
    text = mit_wissen(frage, wissen) if wissen else frage
    raum = max(16, cfg.block_size - max_len - 4)
    ids = tok.encode(build_prompt([{"role": "user", "content": text}]),
                     add_special_tokens=False).ids[-raum:]
    quelle = tok.encode(wissen, add_special_tokens=False).ids
    if len(quelle) < 2:
        return ""
    eos = tok.token_to_id("</s>")
    max_len = min(max_len, len(quelle))

    # Mit leerem Speicher anstossen: ohne uebergebenen Zwischenspeicher legt
    # das Modell gar keinen an, und dann gibt es nichts zu vervielfachen.
    leer = [(None, None)] * cfg.n_layer
    logits, kv = netz(torch.tensor([ids], device=device), kv_caches=leer)

    starts = list(range(len(quelle)))
    B = len(starts)
    kvb = [(k.expand(B, -1, -1, -1).contiguous(),
            v.expand(B, -1, -1, -1).contiguous()) for k, v in kv]
    lp = torch.log_softmax(logits[0, -1], dim=-1).expand(B, -1)

    summe = torch.zeros(B, device=device)
    gueltig = torch.ones(B, dtype=torch.bool, device=device)
    bestes, beste = -1e9, []
    for t in range(max_len):
        stelle = torch.tensor([min(p + t, len(quelle) - 1) for p in starts],
                              device=device)
        gueltig &= torch.tensor([p + t < len(quelle) for p in starts],
                                device=device)
        naechstes = torch.tensor([quelle[int(s)] for s in stelle], device=device)
        summe = summe + lp.gather(1, naechstes[:, None]).squeeze(1)
        lg, kvb = netz(naechstes[:, None], kv_caches=kvb)
        lp = torch.log_softmax(lg[:, -1], dim=-1)
        note = (summe + lp[:, eos]) / (t + 2) ** alpha
        note = note.masked_fill(~gueltig, -1e9)
        i = int(note.argmax())
        if float(note[i]) > bestes:
            bestes = float(note[i])
            beste = quelle[starts[i]:starts[i] + t + 1]
    return tok.decode(beste).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frage", nargs="*")
    ap.add_argument("--ckpt", default="ckpt_8mb/sft.pt")
    ap.add_argument("--tokenizer", default="tokenizer/de_bpe.json")
    ap.add_argument("--wissen", default="wiki")
    ap.add_argument("--fragen", action="store_true")
    ap.add_argument("--ohne-wissen", action="store_true",
                    help="zum Vergleich: dieselbe Frage ohne Absatz")
    ap.add_argument("--absaetze", type=int, default=2,
                    help="wie viele Absätze von der Karte in den Prompt")
    ap.add_argument("--zeichen", type=int, default=1000,
                    help="Zeichenbudget für das Wissen im Prompt")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    from tokenizers import Tokenizer
    device = pick_device(args.device)
    print(f"\n  {GRAU}lade Modell und Index ...{AUS}", flush=True)
    t0 = time.time()
    netz, cfg = lade(args.ckpt, device)
    tok = Tokenizer.from_file(args.tokenizer)
    wb = Wissensbasis(args.wissen)
    print(f"  {GRAU}{wb.n:,} Absätze, bereit in {time.time()-t0:.1f} s{AUS}\n")

    fragen = FRAGEN if args.fragen or not args.frage else [" ".join(args.frage)]
    for frage in fragen:
        t = time.time()
        absatz = (None if args.ohne_wissen else
                  wb.bester_text(frage, args.zeichen, absaetze=args.absaetze))
        t_suche = time.time() - t
        t = time.time()
        antwort = antworte(netz, tok, cfg, device, frage, absatz)
        t_modell = time.time() - t
        print(f"  {FETT}{frage}{AUS}")
        if absatz:
            print(f"  {GRAU}Karte : {absatz[:150]}{'...' if len(absatz)>150 else ''}{AUS}")
        print(f"  {GRUEN}Antwort{AUS}: {antwort[:200]}")
        print(f"  {GRAU}Suche {t_suche*1000:.0f} ms, Modell {t_modell:.1f} s{AUS}\n")


if __name__ == "__main__":
    main()
