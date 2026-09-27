"""Der Stimmen-Schueler: drei kleine Netze nach sanoTTS.

Aufbau wie in arXiv 2608.21378, mit einer bewussten Vereinfachung:

    Dauer      Phonem-IDs            -> wie viele Rahmen je Phonem     36k
    Klang      IDs + Dauern          -> 40 Werte je Rahmen            200k
    Dekoder    40 Werte je Rahmen    -> Betrag und Phase -> Ton       331k

Die Vereinfachung betrifft die mittlere Stufe. sanoTTS laesst den Schueler
eine gelernte 40er-Darstellung nachahmen, die aus den 192 Kanaelen des
Lehrers zusammengezogen wird - dafuer braucht man Zugriff auf das Innere des
Lehrers. Wir haben nur seinen Ton, also nehmen wir stattdessen das
**Mel-Spektrum mit 40 Baendern**. Das ist dieselbe Groesse, gut verstanden
und aus dem Ton berechenbar.

Der Dekoder erzeugt keinen Ton Abtastwert fuer Abtastwert, sondern Betrag und
Phase je Rahmen und laesst eine inverse Fourier-Transformation den Rest
machen. Genau das macht ihn billig genug fuer den Chip: 86 Rahmen je Sekunde
statt 22.050 Abtastwerte.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

RATE, FFT, SPRUNG, MEL = 22050, 1024, 256, 40


class Block(nn.Module):
    """Faltung mit Abkuerzung - dasselbe Muster wie im Erkenner."""

    def __init__(self, kanaele, kern=5):
        super().__init__()
        rand = (kern - 1) // 2
        self.c1 = nn.Conv1d(kanaele, kanaele, kern, padding=rand)
        self.n1 = nn.GroupNorm(1, kanaele)
        self.c2 = nn.Conv1d(kanaele, kanaele, kern, padding=rand)
        self.n2 = nn.GroupNorm(1, kanaele)

    def forward(self, x):
        h = F.gelu(self.n1(self.c1(x)))
        return F.gelu(x + self.n2(self.c2(h)))


class Dauer(nn.Module):
    """Wie lange dauert jedes Phonem? Vorhersage im Logarithmus.

    Im Logarithmus, weil Dauern schief verteilt sind: die meisten Phoneme
    bekommen ein bis drei Rahmen, wenige aber zwanzig. Ein Fehler von einem
    Rahmen wiegt bei einem kurzen Laut schwerer als bei einem langen.
    """

    def __init__(self, n_phon=152, breite=32, bloecke=3):
        super().__init__()
        self.emb = nn.Embedding(n_phon, breite)
        self.bloecke = nn.Sequential(*[Block(breite, 5) for _ in range(bloecke)])
        self.aus = nn.Conv1d(breite, 1, 1)

    def forward(self, ids):
        x = self.emb(ids).transpose(1, 2)
        return self.aus(self.bloecke(x)).squeeze(1)          # log(Dauer)


def dehne(x, dauern):
    """Je Phonem so viele Rahmen erzeugen, wie die Dauer sagt."""
    raus = []
    for b in range(x.shape[0]):
        teile = [x[b, :, i:i+1].expand(-1, int(d))
                 for i, d in enumerate(dauern[b]) if d > 0]
        raus.append(torch.cat(teile, dim=1) if teile else x[b, :, :1])
    n = max(t.shape[1] for t in raus)
    return torch.stack([F.pad(t, (0, n - t.shape[1])) for t in raus])


class Klang(nn.Module):
    """Phoneme und Dauern -> Mel-Spektrum, 40 Baender je Rahmen.

    Zwei Stufen: erst auf Phonemebene, wo der Zusammenhang im Wort zaehlt,
    dann auf Rahmenebene, wo der zeitliche Verlauf zaehlt. Dazwischen wird
    gedehnt - jedes Phonem bekommt so viele Rahmen, wie das Dauermodell sagt.
    """

    def __init__(self, n_phon=152, breite=48, token=3, rahmen=5):
        super().__init__()
        self.emb = nn.Embedding(n_phon, breite)
        self.token = nn.Sequential(*[Block(breite, 5) for _ in range(token)])
        self.rahmen = nn.Sequential(*[Block(breite, 5) for _ in range(rahmen)])
        self.aus = nn.Conv1d(breite, MEL, 1)

    def forward(self, ids, dauern):
        x = self.token(self.emb(ids).transpose(1, 2))
        return self.aus(self.rahmen(dehne(x, dauern)))


class EngBlock(nn.Module):
    """Tiefenweise Faltung, dann Aufweitung - die Bauform aus saanoTTS.

    Die volle Faltung 76x76x7 kostet 40.432 Gewichte und vermischt Kanaele
    und Zeit in einem Schritt. Hier laeuft beides getrennt: erst eine Faltung
    je Kanal ueber die Zeit (532 Gewichte), dann zwei punktweise Faltungen
    ueber eine Aufweitung auf 304 Kanaele (46.208). Zusammen 46.740 statt
    80.864 fuer zwei volle Faltungen - und die Aufweitung schafft Kapazitaet
    da, wo die volle Faltung nur Gewichte haeuft.

    Der Ausgang startet auf null, der Block ist zu Beginn also die Identitaet.
    Das macht tiefe Restnetze von Anfang an ruhig.
    """

    def __init__(self, kanaele, kern=7, weite=4):
        super().__init__()
        self.tief = nn.Conv1d(kanaele, kanaele, kern, padding=(kern - 1) // 2,
                              groups=kanaele)
        self.norm = nn.GroupNorm(1, kanaele)
        self.auf = nn.Conv1d(kanaele, kanaele * weite, 1)
        self.ab = nn.Conv1d(kanaele * weite, kanaele, 1)
        nn.init.zeros_(self.ab.weight)
        nn.init.zeros_(self.ab.bias)

    def forward(self, x):
        return x + self.ab(F.gelu(self.auf(self.norm(self.tief(x)))))


class Dekoder(nn.Module):
    """Mel -> Betrag und Phase -> Ton, ueber inverse Fourier-Transformation.

    Der Ausgang sind 513 Betraege und zweimal 513 Werte fuer die Phase (Real-
    und Imaginaerteil, damit sie stetig bleibt). Daraus baut die inverse
    Transformation die Wellenform. Auf dem Chip kostet das 28 Millionen
    Multiplikationen je Sekunde Ton statt einer Wellenformerzeugung.
    """

    def __init__(self, breite=76, bloecke=5, weite=4, kern=7):
        super().__init__()
        self.ein = nn.Conv1d(MEL, breite, 7, padding=3)
        self.bloecke = nn.Sequential(
            *[EngBlock(breite, kern, weite) for _ in range(bloecke)])
        self.aus = nn.Conv1d(breite, 3 * (FFT // 2 + 1), 1)
        self.register_buffer("fenster", torch.hann_window(FFT))

    def forward(self, mel):
        h = self.aus(self.bloecke(F.gelu(self.ein(mel))))
        n = FFT // 2 + 1
        betrag = torch.exp(torch.clamp(h[:, :n], max=8.0))
        re, im = h[:, n:2*n], h[:, 2*n:]
        norm = torch.sqrt(re ** 2 + im ** 2) + 1e-6
        spektrum = torch.complex(betrag * re / norm, betrag * im / norm)
        return torch.istft(spektrum, FFT, SPRUNG, window=self.fenster,
                           center=True, return_complex=False)


def mel_bank(n_mel=MEL, fft=FFT, rate=RATE):
    """Dreiecksfilter auf der Mel-Skala - wie in asr_daten.py."""
    def zu_mel(f): return 2595 * np.log10(1 + f / 700)
    def zu_hz(m): return 700 * (10 ** (m / 2595) - 1)
    punkte = zu_hz(np.linspace(zu_mel(0), zu_mel(rate / 2), n_mel + 2))
    bins = np.floor((fft + 1) * punkte / rate).astype(int)
    bank = np.zeros((n_mel, fft // 2 + 1), dtype=np.float32)
    for i in range(n_mel):
        a, m, b = bins[i], bins[i+1], bins[i+2]
        for k in range(a, m):
            bank[i, k] = (k - a) / max(m - a, 1)
        for k in range(m, b):
            bank[i, k] = (b - k) / max(b - m, 1)
    return torch.from_numpy(bank)


def zu_mel(ton, bank, fenster):
    """Wellenform -> log-Mel, dieselbe Rechnung wie beim Training des Klangs."""
    s = torch.stft(ton, FFT, SPRUNG, window=fenster, center=True,
                   return_complex=True).abs() ** 2
    return torch.log(bank.to(s.device) @ s + 1e-6)


def groesse(netz):
    n = sum(p.numel() for p in netz.parameters())
    return n, n / 1e6, n / 1e6                     # Parameter, MB als int8


# ------------------------------------------------------------------ Gegenspieler

def _wn(schicht):
    """Gewichtsnormierung, falls die Torch-Fassung sie so kennt."""
    try:
        return nn.utils.parametrizations.weight_norm(schicht)
    except (AttributeError, RuntimeError):
        return schicht


class EinKritiker(nn.Module):
    """Ein Gegenspieler fuer eine Fensterbreite - er sieht das Spektrogramm.

    Warum ueberhaupt einer: Der Betragsverlust allein ist **phasenblind**.
    Zwei Toene mit gleichem Betragsspektrum und verschiedener Phase sind fuer
    ihn gleich gut und klingen doch voellig verschieden - das ist die Ursache
    des blechernen, sprunghaften Klangs. Unser Dekoder gibt 1.026 Phasenwerte
    je Rahmen aus, die bisher **kein Verlust je angesehen hat**.

    Warum auf dem Spektrogramm und nicht auf der Welle: Die Literatur misst
    fuer die Spektrogramm-Fassung MOS 4,21 gegen 4,13 und ein Viertel der
    Trainingszeit, weil die Eingabe schon um den Sprung verkuerzt ist.
    """

    def __init__(self, fft, kanaele=32):
        super().__init__()
        self.fft, self.sprung = fft, fft // 4
        self.register_buffer("fenster", torch.hann_window(fft))
        k = kanaele
        self.schichten = nn.ModuleList([
            _wn(nn.Conv2d(1, k, (3, 9), padding=(1, 4))),
            _wn(nn.Conv2d(k, k, (3, 9), stride=(1, 2), padding=(1, 4))),
            _wn(nn.Conv2d(k, k, (3, 9), stride=(1, 2), padding=(1, 4))),
            _wn(nn.Conv2d(k, k, (3, 3), padding=(1, 1))),
        ])
        self.aus = _wn(nn.Conv2d(k, 1, (3, 3), padding=(1, 1)))

    def forward(self, ton):
        s = torch.stft(ton, self.fft, self.sprung, window=self.fenster,
                       return_complex=True).abs()
        h = torch.log(s + 1e-5).unsqueeze(1)
        merkmale = []
        for schicht in self.schichten:
            h = F.leaky_relu(schicht(h), 0.1)
            merkmale.append(h)
        return self.aus(h), merkmale


class Kritiker(nn.Module):
    """Drei Aufloesungen zusammen - grob, mittel, fein.

    Dieselben Fensterbreiten wie im Betragsverlust, damit beide dasselbe
    Grob-und-Fein im Blick haben. Der Kritiker laeuft nur im Training und
    kommt nie auf das Geraet; seine Groesse ist deshalb gleichgueltig.
    """

    def __init__(self, groessen=(512, 1024, 2048), kanaele=32):
        super().__init__()
        self.teile = nn.ModuleList([EinKritiker(g, kanaele) for g in groessen])

    def forward(self, ton):
        werte, merkmale = [], []
        for teil in self.teile:
            w, m = teil(ton)
            werte.append(w)
            merkmale.append(m)
        return werte, merkmale


def kritiker_verlust(echt, falsch):
    """LSGAN: der Kritiker soll echt auf 1 und falsch auf 0 ziehen."""
    v = 0.0
    for e, f in zip(echt, falsch):
        v = v + ((e - 1.0) ** 2).mean() + (f ** 2).mean()
    return v / len(echt)


def erzeuger_verlust(falsch):
    """Der Erzeuger will, dass der Kritiker seine Ausgabe fuer echt haelt."""
    return sum(((f - 1.0) ** 2).mean() for f in falsch) / len(falsch)


def merkmal_verlust(m_echt, m_falsch):
    """Nicht nur das Urteil zaehlt, sondern auch der Weg dahin.

    Der Erzeuger soll im Kritiker dieselben Zwischenbilder ausloesen wie echte
    Sprache. Das ist der stabilisierende Teil des Gegenspiels - ohne ihn kippt
    LSGAN gern in Tonhoehensprunge.
    """
    v, n = 0.0, 0
    for a, b in zip(m_echt, m_falsch):
        for x, y in zip(a, b):
            v = v + F.l1_loss(y, x.detach())
            n += 1
    return v / max(n, 1)

