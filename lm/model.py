"""Kleiner deutscher Decoder-Transformer (Llama-Stil, bewusst einfach gehalten,
damit die Inferenz spaeter in C/C++ auf dem ESP32-P4 nachgebaut werden kann).

Bausteine: RMSNorm, RoPE, Multi-Head-Attention ohne Bias, SwiGLU-MLP,
gemeinsame Gewichte fuer Embedding und Ausgabe (weight tying).
"""

import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab_size: int = 8192
    n_layer: int = 8
    n_head: int = 6
    d_model: int = 192
    d_ff: int = 512          # wird auf ein Vielfaches von 32 gerundet
    block_size: int = 256    # Kontextlaenge in Token
    dropout: float = 0.0
    rope_theta: float = 10000.0
    # Schichten mehrfach durchlaufen (ALBERT-Trick): n_layer Durchgaenge,
    # aber nur n_layer//geteilt eigene Gewichtssaetze. Spart Speicher,
    # kostet Rechenzeit - auf dem Chip der richtige Tausch.
    geteilt: int = 1
    # Schluessel und Werte auf weniger Koepfe teilen (GQA/MQA). 0 = wie n_head.
    # Der KV-Cache schrumpft im selben Verhaeltnis - auf dem Chip ist er
    # groesser als das Modell selbst.
    n_kv_head: int = 0
    # Nachschlagetabelle je Schicht (Per-Layer Embeddings). 0 = aus.
    # Gehoert auf die SD-Karte: je Token nur n_layer x d_flash Werte, und
    # zwar zusammenhaengend - ein einziger Lesevorgang deckt alle Schichten.
    # Die Adresse ist die Token-Nummer, also schon bekannt, bevor die erste
    # Schicht rechnet: die Karte laesst sich waehrend des vorigen Tokens lesen.
    d_flash: int = 0
    # Tor (Gemma 3n): das Modell entscheidet je Token und Schicht selbst, wie
    # stark die Tabelle einfliesst. Kostet n_layer x d_model x d_flash Gewichte
    # im schnellen Speicher - bei 16x256x64 sind das 0,26M, also nichts.
    flash_tor: bool = False
    # Kontextanteil (Gemma 3n): zusaetzlich zur reinen Nachschlagetabelle ein
    # aus der Worteinbettung berechneter Teil. Ohne ihn haengt der Beitrag nur
    # an der Token-Nummer, nie am Satz drumherum.
    flash_kontext: bool = False
    # Abfragen und Schluessel normieren, bevor sie verrechnet werden. Kostet
    # 2*head_dim Werte je Schicht und bringt bei ternaeren Gewichten 8-12 %:
    # dort werden die Aufmerksamkeitswerte grob und laufen leicht in Extreme.
    qk_norm: bool = False
    # Werte der ersten Schicht an alle spaeteren durchreichen (ResFormer),
    # Mischung je Schicht gelernt. Zehn Zahlen, nochmal rund 3 %.
    wert_residuum: bool = False

    @property
    def kv_heads(self) -> int:
        return self.n_kv_head or self.n_head

    @property
    def head_dim(self) -> int:
        assert self.d_model % self.n_head == 0, "d_model muss durch n_head teilbar sein"
        return self.d_model // self.n_head


# Fertige Groessen. n_params = tatsaechlich gezaehlt (siehe `python -m lm.model`).
PRESETS = {
    # ---- Formvergleich bei gleicher Parameterzahl (~12,85M, V=8192) ----
    # Die Literatur findet keinen weichen Kompromiss zwischen Tiefe und
    # Breite, sondern zwei getrennte Gueteklassen mit 6 Punkten Abstand
    # (12Lx512, 32Lx384, 64Lx256 oben; 4Lx768, 16Lx448, 24Lx384 unten).
    # Gemessen wurde das bei 70M auf englischen Benchmarks, die ein 13M-
    # Modell gar nicht loesen kann - also messen wir es selbst nach.
    "form_4L": Config(vocab_size=8192, n_layer=4, n_head=8, d_model=464, d_ff=1216, block_size=384, n_kv_head=1),
    "form_6L": Config(vocab_size=8192, n_layer=6, n_head=8, d_model=400, d_ff=1056, block_size=384, n_kv_head=1),
    "form_10L": Config(vocab_size=8192, n_layer=10, n_head=8, d_model=320, d_ff=832, block_size=384, n_kv_head=1),
    "form_16L": Config(vocab_size=8192, n_layer=16, n_head=8, d_model=256, d_ff=672, block_size=384, n_kv_head=1),

    # ---- Der Speicher ausgereizt (19.09.2026) ----
    # Vorgabe: Flash, PSRAM und SD-Karte so weit wie moeglich nutzen.
    #
    # Zwei Irrtuemer von mir, beide durch Nachmessen behoben:
    #  1. Geschaetzt statt gemessen. Ternaere Matrizen werden mit 2,00 Bit
    #     gepackt (vier Trits je Byte), nicht 1,58, und die Worteinbettung
    #     bleibt nicht float16, sondern wird auf --bits quantisiert.
    #     Jetzt wird jede Form wirklich exportiert und die Datei gewogen.
    #  2. Nur das Flash gezaehlt. Gewichte muessen dort nicht liegen: was
    #     nicht hineinpasst, wird beim Start von der SD-Karte ins PSRAM
    #     geladen. Damit tragen BEIDE Speicher Gewichte.
    #
    #   Gewichte 10,39 MB  ->  Flash 8,00 von 8 · PSRAM 6,55 von 8
    #   Kern 35,3 M · SD 46 MB Tabelle, 5.632 B je Token
    #   Einbettung 8 Bit: 0,4 % Loss gemessen (4 Bit waeren 8,5 %)
    #
    # Der KV-Cache belegt bei Kontext 2048 und einem KV-Kopf 2,88 MB -
    # deshalb bleibt im PSRAM knapp 1,5 MB Luft.
    "form_gross": Config(vocab_size=8192, n_layer=22, n_head=11, d_model=352,
                         d_ff=928, block_size=2048, n_kv_head=1,
                         qk_norm=True, wert_residuum=True, d_flash=256),
    "form_22L": Config(vocab_size=8192, n_layer=22, n_head=8, d_model=224, d_ff=576, block_size=384, n_kv_head=1),
    "form_32L": Config(vocab_size=8192, n_layer=32, n_head=8, d_model=192, d_ff=512, block_size=384, n_kv_head=1),

    # ~1.4M - nur zum Testen der Pipeline, laeuft auch auf der CPU
    "tiny":  Config(vocab_size=4096, n_layer=4, n_head=4, d_model=128, d_ff=352, block_size=256),
    # ~5M - das Zielmodell fuer den ersten echten Trainingslauf
    "5m":    Config(vocab_size=8192, n_layer=8, n_head=6, d_model=192, d_ff=512, block_size=256),
    # ~10M - Vergleichsgroesse: doppelt so viel Kapazitaet wie das 5M
    "10m":   Config(vocab_size=8192, n_layer=10, n_head=8, d_model=256, d_ff=672, block_size=256),
    # ~8M - Referenzgroesse aus dem Projektplan (cardputer-ai)
    "8m":    Config(vocab_size=8192, n_layer=8, n_head=8, d_model=256, d_ff=672, block_size=256),
    # ~15M
    "15m":   Config(vocab_size=8192, n_layer=10, n_head=8, d_model=320, d_ff=864, block_size=256),
    # Dieselben Gewichte wie 15m, aber doppelt durchlaufen: 20 Durchgaenge
    # ueber 10 eigene Schichten (--teilen 2). Kostet Rechenzeit statt Speicher -
    # der richtige Tausch, wenn Antwortzeit keine Rolle spielt.
    "15m_tief": Config(vocab_size=8192, n_layer=20, n_head=8, d_model=320, d_ff=864, block_size=256),
    # ~47M - der grosse Lauf: 16 Schichten, 512 breit. Mit einem KV-Kopf und
    # Flash-Tabelle gedacht (--kv-head 1 --flash 64), dann 10,7 MB ternaer
    # plus 1,05 MB Cache - passt dreifach in die 32 MB PSRAM des P4.
    "50m":   Config(vocab_size=8192, n_layer=16, n_head=8, d_model=512, d_ff=1376, block_size=256),
    # ~30M - Zielgroesse laut Projektplan
    "30m":   Config(vocab_size=8192, n_layer=12, n_head=8, d_model=448, d_ff=1216, block_size=256),
}


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.to(dtype)) * self.weight


def build_rope_cache(block_size: int, head_dim: int, theta: float, device=None):
    """cos/sin Tabellen fuer Rotary Position Embeddings."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(block_size, device=device).float()
    freqs = torch.outer(t, inv_freq)          # (T, head_dim/2)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x, cos, sin):
    # x: (B, n_head, T, head_dim)
    T = x.size(-2)
    cos = cos[:T].view(1, 1, T, -1)
    sin = sin[:T].view(1, 1, T, -1)
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class Attention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.n_head = cfg.n_head
        self.n_kv = cfg.kv_heads
        assert cfg.n_head % self.n_kv == 0, "n_head muss durch n_kv_head teilbar sein"
        self.head_dim = cfg.head_dim
        # Eine Matrix fuer q, k, v. Bei n_kv == n_head ist die Form identisch
        # zu frueher, also lassen sich alte Checkpoints weiterhin laden.
        self.qkv = nn.Linear(cfg.d_model, (cfg.n_head + 2 * self.n_kv) * cfg.head_dim,
                             bias=False)
        if cfg.qk_norm:
            self.q_norm = nn.Parameter(torch.ones(cfg.head_dim))
            self.k_norm = nn.Parameter(torch.ones(cfg.head_dim))
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x, cos, sin, kv_cache=None, v_erste=None, mix=None):
        B, T, C = x.shape
        nq, nk = self.n_head * self.head_dim, self.n_kv * self.head_dim
        q, k, v = self.qkv(x).split([nq, nk, nk], dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_kv, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_kv, self.head_dim).transpose(1, 2)

        if kv_cache is None:
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)
            causal = True
        else:
            # Generierung Token fuer Token: Position = bisherige Cache-Laenge
            past_k, past_v = kv_cache
            pos = past_k.size(2) if past_k is not None else 0
            q = apply_rope(q, cos[pos:pos + T], sin[pos:pos + T])
            k = apply_rope(k, cos[pos:pos + T], sin[pos:pos + T])
            if past_k is not None:
                k = torch.cat([past_k, k], dim=2)
                v = torch.cat([past_v, v], dim=2)
            kv_cache = (k, v)
            causal = T > 1

        if hasattr(self, "q_norm"):
            # nach RoPE, vor dem Skalarprodukt
            q = F.rms_norm(q, (self.head_dim,), self.q_norm)
            k = F.rms_norm(k, (self.head_dim,), self.k_norm)
        if mix is not None and v_erste is not None:
            t = torch.sigmoid(mix)
            v = (1 - t) * v + t * v_erste
        neue_v = v
        if self.n_kv != self.n_head:
            # erst NACH dem Zwischenspeichern aufblasen - gespeichert wird klein
            wdh = self.n_head // self.n_kv
            k = k.repeat_interleave(wdh, dim=1)
            v = v.repeat_interleave(wdh, dim=1)

        y = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=causal
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y), kv_cache, neue_v


class MLP(nn.Module):
    """SwiGLU: (silu(W1 x) * W3 x) W2"""

    def __init__(self, cfg: Config):
        super().__init__()
        hidden = 32 * ((cfg.d_ff + 31) // 32)
        self.w1 = nn.Linear(cfg.d_model, hidden, bias=False)
        self.w3 = nn.Linear(cfg.d_model, hidden, bias=False)
        self.w2 = nn.Linear(hidden, cfg.d_model, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x, cos, sin, kv_cache=None, v_erste=None, mix=None):
        h, kv_cache, v = self.attn(self.norm1(x), cos, sin, kv_cache, v_erste, mix)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x, kv_cache, v


class TinyLM(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        assert cfg.n_layer % cfg.geteilt == 0, "n_layer muss durch geteilt teilbar sein"
        # Bei geteilt>1 stehen dieselben Block-Objekte mehrfach in der Liste.
        # PyTorch entdoppelt in .parameters(), also zaehlt und optimiert es
        # jeden Satz genau einmal - der KV-Cache bleibt pro Durchgang eigen.
        eigene = [Block(cfg) for _ in range(cfg.n_layer // cfg.geteilt)]
        self.blocks = nn.ModuleList(eigene * cfg.geteilt)
        if cfg.d_flash:
            # die grosse Tabelle - gehoert spaeter auf die SD-Karte
            self.flash_tab = nn.Embedding(cfg.vocab_size, cfg.n_layer * cfg.d_flash)
            # die kleinen Projektionen - bleiben im schnellen Speicher
            self.flash_proj = nn.ModuleList(
                [nn.Linear(cfg.d_flash, cfg.d_model, bias=False)
                 for _ in range(cfg.n_layer)])
            if cfg.flash_tor:
                self.flash_tor_proj = nn.ModuleList(
                    [nn.Linear(cfg.d_model, cfg.d_flash, bias=False)
                     for _ in range(cfg.n_layer)])
            if cfg.flash_kontext:
                self.flash_ktx = nn.Linear(
                    cfg.d_model, cfg.n_layer * cfg.d_flash, bias=False)
        self.norm_f = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # weight tying spart ~1.5M Parameter

        if cfg.wert_residuum:
            # gelernte Mischung je Schicht: wie stark zaehlt die erste Schicht?
            self.wert_mix = nn.Parameter(torch.zeros(cfg.n_layer))

        cos, sin = build_rope_cache(cfg.block_size, cfg.head_dim, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # Residual-Projektionen kleiner initialisieren (GPT-2 Rezept)
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("w2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_emb.weight.numel()
        return n

    def num_params_kern(self) -> int:
        """Parameter, die wirklich ins Flash muessen.

        Die Tabelle aus `d_flash` zaehlt nicht mit: sie wird je Token einmal
        von der SD-Karte gelesen und danach verworfen. Fuer das 8-MB-Budget
        ist allein diese Zahl massgeblich.
        """
        n = self.num_params()
        if self.cfg.d_flash:
            n -= self.flash_tab.weight.numel()
        return n

    def speicher_bericht(self, bits: float = 1.58) -> dict:
        """Was belegt wo - in Byte, bei ternaeren Gewichten."""
        kern = self.num_params_kern()
        tab = self.flash_tab.weight.numel() if self.cfg.d_flash else 0
        return {
            "kern_parameter": kern,
            "flash_byte": int(kern * bits / 8),
            "tabelle_parameter": tab,
            "sd_byte": tab,                       # int8 je Wert
            "sd_je_token_byte": self.cfg.n_layer * self.cfg.d_flash if tab else 0,
        }

    def forward(self, idx, targets=None, kv_caches=None):
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"Sequenz {T} > block_size {self.cfg.block_size}"
        x = self.drop(self.tok_emb(idx))
        ple = (self.flash_tab(idx).view(B, T, self.cfg.n_layer, self.cfg.d_flash)
               if self.cfg.d_flash else None)
        if ple is not None and self.cfg.flash_kontext:
            # Der Kontextanteil kommt aus der Worteinbettung, also vor der
            # ersten Schicht - sonst waere er fuer jede Schicht ein anderer.
            ple = ple + self.flash_ktx(x).view(B, T, self.cfg.n_layer, self.cfg.d_flash)
        new_caches, v_erste = [], None
        for i, block in enumerate(self.blocks):
            if ple is not None:
                p = ple[:, :, i, :]
                if self.cfg.flash_tor:
                    p = p * torch.sigmoid(self.flash_tor_proj[i](x))
                x = x + self.flash_proj[i](p)
            cache = kv_caches[i] if kv_caches is not None else None
            mix = (self.wert_mix[i] if self.cfg.wert_residuum and i > 0 else None)
            x, cache, v = block(x, self.rope_cos, self.rope_sin, cache, v_erste, mix)
            if self.cfg.wert_residuum and v_erste is None:
                v_erste = v
            new_caches.append(cache)
        x = self.norm_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100
            )
            return logits, loss
        # bei der Generierung reicht das letzte Token
        logits = self.lm_head(x[:, [-1], :])
        return logits, (new_caches if kv_caches is not None else None)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=100, temperature=0.8, top_k=40,
                 top_p=0.95, repetition_penalty=1.1, eos_id=None):
        self.eval()
        # Wichtig: (None, None) statt None. In Attention.forward bedeutet ein
        # blankes None "ohne Cache rechnen" - dann wuerde der Cache nie gefuellt
        # und das Modell saehe ab dem zweiten Token nur noch das letzte Token
        # ohne jede Vorgeschichte. Das Tupel schaltet den Cache-Zweig ein.
        leer = [(None, None)] * self.cfg.n_layer
        kv_caches = list(leer)
        cur = idx
        generated = []
        for _ in range(max_new_tokens):
            # Kontextfenster einhalten
            if cur.size(1) > self.cfg.block_size:
                cur = cur[:, -self.cfg.block_size:]
                kv_caches = list(leer)
            logits, kv_caches = self(cur, kv_caches=kv_caches)
            logits = logits[:, -1, :].float()

            if repetition_penalty != 1.0 and generated:
                # Das Vorzeichen ist entscheidend: ein negatives Logit wird durch
                # Teilen GROESSER, die Strafe wuerde die Wiederholung also belohnen.
                # Bei diesem Modell sind rund 87 % aller Logits negativ - ohne
                # diese Fallunterscheidung landet die Ausgabe in Endlosschleifen.
                letzte = torch.tensor(sorted(set(generated[-64:])), device=logits.device)
                werte = logits[:, letzte]
                logits[:, letzte] = torch.where(werte > 0,
                                                werte / repetition_penalty,
                                                werte * repetition_penalty)

            if temperature <= 0:
                nxt = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k:
                    kth = torch.topk(logits, min(top_k, logits.size(-1)))[0][:, [-1]]
                    logits[logits < kth] = -float("inf")
                probs = F.softmax(logits, dim=-1)
                if top_p and top_p < 1.0:
                    sp, si = torch.sort(probs, descending=True, dim=-1)
                    cum = sp.cumsum(dim=-1)
                    sp[cum - sp > top_p] = 0.0
                    sp = sp / sp.sum(dim=-1, keepdim=True)
                    nxt = si.gather(-1, torch.multinomial(sp, 1))
                else:
                    nxt = torch.multinomial(probs, 1)

            tok = int(nxt.item())
            generated.append(tok)
            yield tok
            if eos_id is not None and tok == eos_id:
                break
            cur = nxt  # ab jetzt nur noch das neue Token, Rest steckt im KV-Cache


def build(preset: str = "5m", **overrides) -> TinyLM:
    cfg = Config(**{**asdict(PRESETS[preset]), **overrides})
    return TinyLM(cfg)


if __name__ == "__main__":
    print(f"{'preset':8} {'params':>12} {'davon Embedding':>16} {'Q4 roh (MB)':>12}")
    for name in PRESETS:
        m = build(name)
        n = m.num_params()
        emb = m.tok_emb.weight.numel()
        print(f"{name:8} {n:12,} {emb:16,} {n * 0.5 / 1e6:12.1f}")
