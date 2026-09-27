"""Ternäres Training nach BitNet-Art: Gewichte nur -1, 0, +1.

Die Idee, die dein Freund hatte, ist Forschungsfront. Statt jedes Gewicht als
Zahl mit 16 oder 32 Bit zu speichern, darf es nur drei Werte annehmen. Das
klingt nach massivem Informationsverlust - funktioniert bei grossen Modellen
aber verblueffend gut, weil das Netz *waehrend* des Trainings lernt, mit der
Einschraenkung zu leben.

Der entscheidende Unterschied zur nachtraeglichen Quantisierung:

    nachtraeglich   fertig trainiertes Gewicht 0.42  ->  1     (Fehler 0.58)
    im Training     Netz weiss von Anfang an, dass nur -1,0,1 geht
                    und richtet alle anderen Gewichte danach aus

Technisch geht das mit einem Trick (straight-through estimator): Vorwaerts
rechnet das Netz mit den ternaeren Werten, rueckwaerts fliesst der Gradient
an der Rundung vorbei zu den vollen Gewichten. Die vollen Werte existieren
also nur beim Training, im fertigen Modell bleiben drei Stufen.

Rechenvorteil auf dem ESP32: Eine Multiplikation mit -1, 0 oder +1 ist keine
Multiplikation, sondern ein Vorzeichenwechsel, ein Weglassen oder ein
Durchlassen. Der Chip braucht dafuer keinen Multiplizierer - nur Addierer.

    python ternaer.py vergleich --steps 400
    python ternaer.py nachtraeglich --ckpt ckpt_colab/sft.pt
"""

import argparse
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import model as M


# ------------------------------------------------------------------ Ternär

def ternarisiere(w: torch.Tensor):
    """Gewichtsmatrix -> {-1, 0, +1} mit einer Skala pro Matrix.

    Die Skala ist der mittlere Betrag aller Gewichte. Wer durch sie teilt und
    rundet, landet bei den drei Stufen; Werte nahe null werden zu null - das
    macht die Matrix zusaetzlich spaerlich.
    """
    skala = w.abs().mean().clamp(min=1e-8)
    return (w / skala).round().clamp(-1, 1) * skala


class BitLinear(nn.Linear):
    """Lineare Schicht mit ternaeren Gewichten und vollem Gradienten."""

    def forward(self, x):
        w_t = ternarisiere(self.weight)
        # straight-through: vorwaerts ternaer, rueckwaerts am Runden vorbei
        w = self.weight + (w_t - self.weight).detach()
        return F.linear(x, w, self.bias)


# ------------------------------------------------------- Sherry: 3 von 4 (1,25 Bit)

def sherrisiere(w: torch.Tensor):
    """3:4-Ternaer nach Sherry: je vier Gewichte ist genau eines null.

    Warum ausgerechnet drei von vier: Vier Moeglichkeiten fuer die Lage der Null
    mal acht Vorzeichenmuster ergeben 32 Zustaende - exakt 5 Bit fuer vier
    Gewichte, also 1,25 Bit je Gewicht. Es geht kein Bit verloren, und die
    Bloecke bleiben an Zweierpotenzen ausgerichtet, was den Vektorbefehlen des
    ESP32-P4 entgegenkommt.

        vier freie Trits in ein Byte   2,00 Bit   (unsere bisherige Packung)
        fuenf Trits in ein Byte        1,60 Bit   (braucht eine Division)
        Sherry 3:4                     1,25 Bit   (nur Schieben und Maskieren)

    Der Preis: die Duenne liegt fest bei genau 25 %. Gewichte, die frei ternaer
    auf null fielen, muessen hier ein Vorzeichen annehmen - und umgekehrt.
    Ob das QAT das auffaengt, misst `experimente.py sherry`.
    """
    assert w.numel() % 4 == 0, "Sherry braucht eine durch 4 teilbare Gewichtszahl"
    f = w.reshape(-1, 4)
    # das betragskleinste Gewicht je Viererblock wird geopfert
    idx = f.abs().argmin(dim=1, keepdim=True)
    behalten = torch.ones_like(f).scatter_(1, idx, 0.0)
    # Skala nur aus den drei ueberlebenden Gewichten, nicht aus allen vieren
    skala = ((f.abs() * behalten).sum() / behalten.sum().clamp(min=1.0)).clamp(min=1e-8)
    return (torch.sign(f) * behalten * skala).reshape(w.shape)


class SherryLinear(nn.Linear):
    """Lineare Schicht mit 3:4-ternaeren Gewichten und vollem Gradienten."""

    def forward(self, x):
        w_s = sherrisiere(self.weight)
        w = self.weight + (w_s - self.weight).detach()
        return F.linear(x, w, self.bias)


def zu_sherry(netz: nn.Module, ausser=("lm_head",)):
    return _ersetze(netz, SherryLinear, ausser)


# ------------------------------------------------- lernbare Schwelle (CAT-Q)

class SchwellLinear(nn.Linear):
    """Ternaer, aber die Grenze zwischen null und plusminus eins wird gelernt.

    Die feste Rundung in `ternarisiere` setzt die Grenze bei 0,5 mal dem
    mittleren Betrag. CAT-Q misst, dass eine gelernte Grenze 5 bis 10 % besser
    abschneidet - die richtige Duenne haengt von der Schicht ab.

    Damit die Grenze ueberhaupt einen Gradienten bekommt, laeuft neben dem
    harten Schnitt ein weicher mit: vorwaerts gilt der harte Wert, rueckwaerts
    fliesst der Gradient sowohl am Runden vorbei zum Gewicht (wie bisher) als
    auch durch den weichen Schnitt zur Grenze.
    """

    def einrichten(self, start=0.5, schaerfe=10.0):
        self.schwelle = nn.Parameter(torch.tensor(float(start)))
        self.schaerfe = schaerfe
        return self

    def forward(self, x):
        skala = self.weight.abs().mean().clamp(min=1e-8)
        n = self.weight / skala
        t = self.schwelle.clamp(0.05, 1.5)
        vz = torch.sign(n)
        hart = vz * (n.abs() > t).float() * skala
        weich = vz * torch.sigmoid((n.abs() - t) * self.schaerfe) * skala
        # Wert = harte Stufen; Gradient = STE zum Gewicht + weicher Pfad zur Grenze
        w = (hart.detach()
             + (self.weight - self.weight.detach())
             + (weich - weich.detach()))
        return F.linear(x, w, self.bias)


def zu_schwelle(netz: nn.Module, ausser=("lm_head",), start=0.5):
    ersetzt = _ersetze(netz, SchwellLinear, ausser)
    for m in netz.modules():
        if isinstance(m, SchwellLinear):
            m.einrichten(start)
    return ersetzt


def duenne(netz):
    """Anteil der Gewichte, die tatsaechlich auf null fallen - je Bauart."""
    null = gesamt = 0
    for m in netz.modules():
        if isinstance(m, (BitLinear, SherryLinear, SchwellLinear, SherrySkala)):
            with torch.no_grad():
                if isinstance(m, (SherryLinear, SherrySkala)):
                    w = sherrisiere(m.weight)
                elif isinstance(m, SchwellLinear):
                    n = m.weight / m.weight.abs().mean().clamp(min=1e-8)
                    w = torch.where(n.abs() > m.schwelle.clamp(0.05, 1.5),
                                    torch.sign(n), torch.zeros_like(n))
                else:
                    w = ternarisiere(m.weight)
                null += (w == 0).sum().item()
                gesamt += w.numel()
    return null / max(gesamt, 1)


# ------------------------------------------------------------------ Binär

def binarisiere(w: torch.Tensor):
    """Gewichtsmatrix -> {-1, +1} mit einer Skala je Ausgabekanal.

    Ohne die Null bleibt ein Bit je Gewicht - acht passen in ein Byte, und die
    Multiplikation wird auf dem Chip zu XNOR plus Bitzaehlen. Die Skala je
    Kanal (statt je Matrix wie beim ternaeren Fall) gleicht aus, dass hier
    keine kleinen Gewichte mehr auf null fallen koennen.
    """
    skala = w.abs().mean(dim=1, keepdim=True).clamp(min=1e-8)
    return torch.sign(w + 1e-12) * skala


class BinLinear(nn.Linear):
    """Lineare Schicht mit binaeren Gewichten und vollem Gradienten."""

    def forward(self, x):
        w_b = binarisiere(self.weight)
        w = self.weight + (w_b - self.weight).detach()
        return F.linear(x, w, self.bias)


def zu_binaer(netz: nn.Module, ausser=("lm_head",)):
    """Wie zu_ternaer, nur mit zwei Stufen statt drei."""
    return _ersetze(netz, BinLinear, ausser)


def beschraenke(netz: nn.Module, art, ausser=("lm_head",)):
    """Einstiegspunkt fuer train.py/sft.py.

    art ist 'ternaer', 'sherry', 'binaer' oder None.

    'sherry' gehoert hierher und nicht in den Export: gemessen am 29.08.
    kostet der 3:4-Zwang NACHTRAEGLICH zwoelf Punkte Lesefaehigkeit
    (85,3 auf 73,3 Prozent), als Trainingsart dagegen hoechstens zwei -
    in der Versuchsreihe war er vom freien Ternaeren nicht zu unterscheiden.
    Wer mit 1,25 Bit ausliefern will, muss mit 1,25 Bit trainieren.
    """
    if not art:
        return 0
    bauer = {"ternaer": zu_ternaer, "sherry": zu_sherry, "binaer": zu_binaer}
    if art not in bauer:
        raise ValueError(f"unbekannte Beschraenkung: {art}")
    return bauer[art](netz, ausser)


def _ersetze(netz: nn.Module, klasse, ausser):
    ersetzt = 0
    for name, kind in netz.named_children():
        if isinstance(kind, nn.Linear) and not any(a in name for a in ausser):
            neu = klasse(kind.in_features, kind.out_features,
                         bias=kind.bias is not None)
            neu.weight = kind.weight
            if kind.bias is not None:
                neu.bias = kind.bias
            setattr(netz, name, neu)
            ersetzt += 1
        else:
            ersetzt += _ersetze(kind, klasse, ausser)
    return ersetzt


def zu_ternaer(netz: nn.Module, ausser=("lm_head",)):
    """Alle linearen Schichten ersetzen - bis auf die Ausgabeschicht.

    Warum die Ausgabe ausgenommen bleibt: Sie teilt sich die Gewichte mit dem
    Embedding, und beides ternaer zu machen kostet bei kleinen Modellen
    ueberproportional viel. Genau das berichten auch die Arbeiten zu 1.58 Bit:
    Ein- und Ausgabeschichten bleiben in hoeherer Genauigkeit.
    """
    return _ersetze(netz, BitLinear, ausser)


def statistik(netz):
    """Wie viele Gewichte werden null, plus eins, minus eins?"""
    null = plus = minus = gesamt = 0
    for modul in netz.modules():
        if isinstance(modul, BitLinear):
            w = ternarisiere(modul.weight)
            skala = modul.weight.abs().mean().clamp(min=1e-8)
            stufen = (w / skala).round()
            null += (stufen == 0).sum().item()
            plus += (stufen > 0).sum().item()
            minus += (stufen < 0).sum().item()
            gesamt += stufen.numel()
    return null, plus, minus, gesamt


# ------------------------------------------------------------------ Vergleich

def einen_lauf(preset, steps, ternär, device, daten, batch=32, lr=6e-4):
    from train import Batches

    torch.manual_seed(1234)                 # gleicher Start für beide Läufe
    netz = M.build(preset)
    ersetzt = zu_ternaer(netz) if ternär else 0
    netz.to(device)

    n = sum(p.numel() for p in netz.parameters())
    art = f"ternär ({ersetzt} Schichten)" if ternär else "float"
    print(f"  {art:28} {n:,} Parameter")

    opt = torch.optim.AdamW(netz.parameters(), lr=lr, betas=(0.9, 0.95),
                            weight_decay=0.1)
    plan = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps,
                                               pct_start=0.15)
    verlauf, t0 = [], time.time()
    for schritt in range(steps):
        x, y = daten.get()
        _, verlust = netz(x, y)
        opt.zero_grad(); verlust.backward()
        torch.nn.utils.clip_grad_norm_(netz.parameters(), 1.0)
        opt.step(); plan.step()
        verlauf.append(verlust.item())
        if (schritt + 1) % 50 == 0:
            mittel = sum(verlauf[-50:]) / 50
            print(f"\r    Schritt {schritt+1:>4}/{steps}  Verlust {mittel:6.3f}"
                  f"  ppl {math.exp(min(mittel,20)):7.1f}  "
                  f"{time.time()-t0:5.0f}s", end="", flush=True)
    print()
    return netz, verlauf


def vergleich(args):
    from train import Batches, pick_device

    device = pick_device(args.device)
    daten = Batches("data/train.bin", 256, args.batch, device)
    print(f"\nGleiche Daten, gleicher Startwert, {args.steps} Schritte, "
          f"Preset {args.preset}\n")

    print("1) Volle Genauigkeit als Messlatte")
    _, verlauf_f = einen_lauf(args.preset, args.steps, False, device, daten,
                              args.batch, args.lr)

    preset_t = args.preset_ternaer or args.preset
    print(f"\n2) Ternär von Anfang an ({preset_t})")
    netz_t, verlauf_t = einen_lauf(preset_t, args.steps, True, device, daten,
                                   args.batch, args.lr)

    def mittel(v, n=50):
        return sum(v[-n:]) / min(n, len(v))

    f, t = mittel(verlauf_f), mittel(verlauf_t)
    print(f"\n{'':6}{'Verlust':>10}{'ppl':>10}")
    print(f"  {'float':4}{f:10.3f}{math.exp(min(f,20)):10.1f}")
    print(f"  {'ternär':4}{t:10.3f}{math.exp(min(t,20)):10.1f}")
    print(f"\n  Abstand: {t-f:+.3f} Verlust  ({(math.exp(min(t,20))/math.exp(min(f,20))-1)*100:+.0f} % ppl)")

    import model as MM
    def budget(preset, ternär):
        m = MM.build(preset); emb = m.cfg.vocab_size * m.cfg.d_model
        rest = m.num_params() - emb
        return (emb * 0.5 + rest * (math.log2(3) / 8 if ternär else 0.5)) / 1e6

    print(f"\nSpeicher auf dem Chip (Embedding jeweils Q4):")
    print(f"  {args.preset:6} in Q4      {budget(args.preset, False):5.2f} MB")
    print(f"  {preset_t:6} ternär      {budget(preset_t, True):5.2f} MB")

    null, plus, minus, gesamt = statistik(netz_t)
    print(f"\nVerteilung der ternären Gewichte:")
    print(f"  null   {null/gesamt*100:5.1f} %   (kostet keine Rechenoperation)")
    print(f"  plus   {plus/gesamt*100:5.1f} %")
    print(f"  minus  {minus/gesamt*100:5.1f} %")

    # Speicherbedarf: 1.58 Bit je Gewicht plus eine Skala je Matrix
    ternär_bits = gesamt * math.log2(3)
    print(f"\nSpeicher für die ternären Schichten:")
    print(f"  float32 {gesamt*4/1e6:6.2f} MB")
    print(f"  Q4      {gesamt*0.5/1e6:6.2f} MB")
    print(f"  ternär  {ternär_bits/8/1e6:6.2f} MB   (1,58 Bit je Gewicht)")


def nachtraeglich(args):
    """Fertiges Modell nachtraeglich ternarisieren - zum Vergleich mit QAT."""
    import numpy as np
    from train import pick_device

    device = pick_device(args.device)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    netz = M.TinyLM(M.Config(**ck["config"]))
    netz.load_state_dict(ck["model"])
    netz.to(device).eval()

    d = np.fromfile("data/val.bin", dtype=np.uint16)
    rng = np.random.default_rng(0)
    proben = [torch.from_numpy(d[i:i+257].astype(np.int64))
              for i in rng.integers(0, len(d)-258, 25)]
    x = torch.stack([p[:-1] for p in proben]).to(device)
    y = torch.stack([p[1:] for p in proben]).to(device)

    with torch.no_grad():
        _, vorher = netz(x, y)
        original = {}
        for name, p in netz.named_parameters():
            if p.dim() >= 2 and "tok_emb" not in name:
                original[name] = p.detach().clone()
                p.copy_(ternarisiere(p))
        _, nachher = netz(x, y)

    print(f"\n{args.ckpt}")
    print(f"  vorher  Verlust {vorher.item():.4f}  ppl {math.exp(vorher.item()):8.1f}")
    print(f"  ternär  Verlust {nachher.item():.4f}  ppl {math.exp(min(nachher.item(),20)):8.1f}")
    print(f"  Verschlechterung: {(nachher.item()/vorher.item()-1)*100:+.0f} %")
    print("\nZum Vergleich Q4 an demselben Modell: rund +5 %.")


def main():
    ap = argparse.ArgumentParser()
    unter = ap.add_subparsers(dest="befehl", required=True)

    v = unter.add_parser("vergleich")
    v.add_argument("--preset", default="5m")
    v.add_argument("--preset-ternaer", default=None,
                   help="andere Groesse fuer den ternaeren Lauf - fuer den "
                        "Vergleich bei gleichem Speicher statt gleicher Parameterzahl")
    v.add_argument("--steps", type=int, default=400)
    v.add_argument("--batch", type=int, default=32)
    v.add_argument("--lr", type=float, default=6e-4)
    v.add_argument("--device", default="auto")

    n = unter.add_parser("nachtraeglich")
    n.add_argument("--ckpt", default="ckpt_colab/sft.pt")
    n.add_argument("--device", default="auto")

    args = ap.parse_args()
    if args.befehl == "vergleich":
        vergleich(args)
    else:
        nachtraeglich(args)


if __name__ == "__main__":
    main()


# --------------------------------------------- gelernte Skala (Sherry, binär)

class SkalaMixin:
    """Ein gelernter Faktor auf die Skala.

    Bei Sherry und binaer gibt es keine Schwelle zu lernen - welches Gewicht
    null wird, steht durch den Betrag fest. Was bleibt, ist die Hoehe der
    Stufen. Die feste Regel nimmt den mittleren Betrag; ob der optimal ist,
    misst dieser Faktor.
    """

    def einrichten(self, start=1.0):
        self.skalafaktor = nn.Parameter(torch.tensor(float(start)))
        return self


class SherrySkala(SkalaMixin, nn.Linear):
    def forward(self, x):
        w_s = sherrisiere(self.weight) * self.skalafaktor.clamp(0.2, 5.0)
        w = self.weight + (w_s - self.weight).detach()
        return F.linear(x, w, self.bias)


class BinSkala(SkalaMixin, nn.Linear):
    def forward(self, x):
        w_b = binarisiere(self.weight) * self.skalafaktor.clamp(0.2, 5.0)
        w = self.weight + (w_b - self.weight).detach()
        return F.linear(x, w, self.bias)


def _mit_skala(netz, klasse, ausser):
    n = _ersetze(netz, klasse, ausser)
    for m in netz.modules():
        if isinstance(m, klasse):
            m.einrichten()
    return n


def zu_sherry_skala(netz, ausser=("lm_head",)):
    return _mit_skala(netz, SherrySkala, ausser)


def zu_binaer_skala(netz, ausser=("lm_head",)):
    return _mit_skala(netz, BinSkala, ausser)


# ------------------------------------------------ Faltungen (Spracherkenner)

class BitConv1d(nn.Conv1d):
    """Faltung mit ternaeren Gewichten.

    Der Spracherkenner besteht aus Conv1d statt Linear, deshalb greift
    `zu_ternaer` dort nicht. Die Rechnung ist dieselbe: vorwaerts drei Stufen,
    rueckwaerts am Runden vorbei.

    Eine Skala je Ausgabekanal statt je Matrix - bei Faltungen liegen die
    Betraege der Kanaele weiter auseinander als bei linearen Schichten, und
    die Skala kostet nur wenige Zahlen.
    """

    def forward(self, x):
        w = self.weight
        skala = w.abs().mean(dim=(1, 2), keepdim=True).clamp(min=1e-8)
        w_t = (w / skala).round().clamp(-1, 1) * skala
        w = w + (w_t - w).detach()
        return self._conv_forward(x, w, self.bias)


def zu_ternaer_conv(netz: nn.Module, ausser=("aus",)):
    """Alle Faltungen ersetzen - bis auf die Ausgabeschicht.

    Die letzte 1x1-Faltung erzeugt die Zeichenwahrscheinlichkeiten. Sie
    ternaer zu machen kostet bei 34 Ausgaengen ueberproportional viel, spart
    aber fast nichts - dieselbe Ueberlegung wie beim lm_head des Sprachmodells.
    """
    ersetzt = 0
    for name, kind in netz.named_children():
        if isinstance(kind, nn.Conv1d) and not any(a in name for a in ausser):
            neu = BitConv1d(kind.in_channels, kind.out_channels,
                            kind.kernel_size, stride=kind.stride,
                            padding=kind.padding, dilation=kind.dilation,
                            bias=kind.bias is not None)
            neu.weight = kind.weight
            if kind.bias is not None:
                neu.bias = kind.bias
            setattr(netz, name, neu)
            ersetzt += 1
        else:
            ersetzt += zu_ternaer_conv(kind, ausser)
    return ersetzt
