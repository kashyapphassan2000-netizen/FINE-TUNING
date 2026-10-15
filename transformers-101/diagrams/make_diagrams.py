"""Generate every SVG diagram used in the Transformers 101 guide.

Run: python make_diagrams.py   (writes *.svg next to this file)
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))

C = {
    "blue": "#2563eb", "blue_l": "#dbeafe",
    "purple": "#7c3aed", "purple_l": "#ede9fe",
    "green": "#059669", "green_l": "#d1fae5",
    "orange": "#ea580c", "orange_l": "#ffedd5",
    "red": "#dc2626", "red_l": "#fee2e2",
    "grey": "#475569", "grey_l": "#f1f5f9", "line": "#334155",
    "yellow_l": "#fef9c3", "teal": "#0d9488", "teal_l": "#ccfbf1",
}
FONT = "Inter, 'Segoe UI', Arial, sans-serif"


class SVG:
    def __init__(self, w, h, title=None):
        self.w, self.h, self.parts = w, h, []
        if title:
            self.text(w / 2, 26, title, size=17, weight=700)

    def rect(self, x, y, w, h, fill="#fff", stroke=C["line"], rx=8, sw=1.5, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
                          f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')

    def text(self, x, y, s, size=13, weight=400, color="#0f172a", anchor="middle", italic=False, mono=False):
        fam = "Consolas, 'Courier New', monospace" if mono else FONT
        st = ' font-style="italic"' if italic else ""
        s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        self.parts.append(f'<text x="{x}" y="{y}" font-family="{fam}" font-size="{size}" '
                          f'font-weight="{weight}" fill="{color}" text-anchor="{anchor}"{st}>{s}</text>')

    def box(self, x, y, w, h, label, fill, stroke, size=13, weight=600, sub=None, sub_size=11):
        self.rect(x, y, w, h, fill, stroke)
        if sub:
            self.text(x + w / 2, y + h / 2 - 3, label, size, weight)
            self.text(x + w / 2, y + h / 2 + 13, sub, sub_size, 400, C["grey"])
        else:
            self.text(x + w / 2, y + h / 2 + size * 0.35, label, size, weight)

    def arrow(self, x1, y1, x2, y2, color=C["line"], sw=1.8, dash=None, label=None, lx=0, ly=-6):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        mid = f"m{color[1:]}"
        self.parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
                          f'stroke-width="{sw}" marker-end="url(#{mid})"{d}/>')
        self._markers.add(color)
        if label:
            self.text((x1 + x2) / 2 + lx, (y1 + y2) / 2 + ly, label, 11, 500, C["grey"])

    def path(self, d, color=C["line"], sw=1.8, arrow=True, dash=None, fill="none"):
        da = f' stroke-dasharray="{dash}"' if dash else ""
        m = f' marker-end="url(#m{color[1:]})"' if arrow else ""
        self.parts.append(f'<path d="{d}" fill="{fill}" stroke="{color}" stroke-width="{sw}"{m}{da}/>')
        if arrow:
            self._markers.add(color)

    def circle(self, x, y, r, fill, stroke=C["line"], sw=1.5):
        self.parts.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')

    def grid(self, x, y, n, cell, allowed, labels=None, on=C["green"], off="#e2e8f0", values=None):
        for i in range(n):
            for j in range(n):
                ok = allowed(i, j)
                fill = on if ok else off
                op = 1.0
                if values is not None and ok:
                    op = 0.25 + 0.75 * values[i][j]
                self.parts.append(f'<rect x="{x + j * cell}" y="{y + i * cell}" width="{cell - 2}" '
                                  f'height="{cell - 2}" rx="3" fill="{fill}" fill-opacity="{op:.2f}"/>')
        if labels:
            for k, lab in enumerate(labels):
                self.text(x - 6, y + k * cell + cell / 2 + 4, lab, 11, 500, anchor="end")
                self.parts.append(
                    f'<text x="{x + k * cell + cell / 2}" y="{y - 8}" font-family="{FONT}" font-size="11" '
                    f'font-weight="500" fill="#0f172a" text-anchor="start" '
                    f'transform="rotate(-45 {x + k * cell + cell / 2} {y - 8})">{lab}</text>')

    def save(self, name):
        markers = "".join(
            f'<marker id="m{c[1:]}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
            f'markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>'
            for c in sorted(self._markers))
        svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" '
               f'width="{self.w}" height="{self.h}"><defs>{markers}</defs>'
               f'<rect width="100%" height="100%" fill="#ffffff"/>' + "".join(self.parts) + "</svg>")
        with open(os.path.join(HERE, name), "w", encoding="utf-8") as f:
            f.write(svg)

    _markers = None


def new(w, h, title=None):
    s = SVG.__new__(SVG)
    s._markers = set()
    SVG.__init__(s, w, h, title)
    return s


# ---------------------------------------------------------------- D0 environment map
def env_map():
    s = new(900, 300, "Where you will work in this guide")
    s.box(20, 70, 160, 70, "Your laptop", C["grey_l"], C["grey"], sub="Windows 11 + Chrome")
    s.box(240, 70, 190, 70, "Google Colab", C["orange_l"], C["orange"], sub="notebook + free T4 GPU")
    s.box(500, 50, 170, 55, "Hugging Face Hub", C["yellow_l"], "#ca8a04", sub="pretrained models")
    s.box(500, 120, 170, 55, "GitHub (raw)", C["grey_l"], C["grey"], sub="Tiny Shakespeare data")
    s.box(720, 70, 160, 70, "Google Drive", C["blue_l"], C["blue"], sub="saved notebooks")
    s.box(240, 200, 190, 60, "Optional: VS Code", C["purple_l"], C["purple"], sub="local Python on Windows")
    s.arrow(180, 105, 238, 105, label="browser")
    s.arrow(430, 90, 498, 78, label="download")
    s.arrow(430, 120, 498, 145)
    s.arrow(430, 105, 718, 105, dash="5 4")
    s.text(640, 196, "File > Save a copy in Drive", 11, 500, C["grey"])
    s.arrow(100, 140, 238, 225, dash="5 4")
    s.save("d00_environment.svg")


# ---------------------------------------------------------------- D1 big picture
def big_picture():
    s = new(1120, 330, "A Transformer, end to end (decoder-only example)")
    y = 120
    items = [
        ("Text", '"the cat sat"', C["grey_l"], C["grey"]),
        ("Tokenizer", "split + IDs", C["grey_l"], C["grey"]),
        ("Token IDs", "[464, 3797, 3332]", C["grey_l"], C["grey"]),
        ("Embedding", "ID -> vector", C["blue_l"], C["blue"]),
        ("+ Position", "where is it?", C["teal_l"], C["teal"]),
    ]
    x = 15
    for i, (a, b, f, st) in enumerate(items):
        s.box(x, y, 118, 64, a, f, st, sub=b)
        if i < len(items) - 1:
            s.arrow(x + 118, y + 32, x + 136, y + 32)
        x += 136
    s.arrow(x - 18, y + 32, x + 4, y + 32)
    s.rect(x + 6, 60, 170, 190, C["purple_l"], C["purple"], rx=12)
    s.text(x + 91, 82, "N x Transformer block", 13, 700, C["purple"])
    s.box(x + 22, 100, 138, 48, "Masked multi-head", "#fff", C["purple"], size=12, sub="self-attention")
    s.box(x + 22, 170, 138, 48, "Feed-forward", "#fff", C["green"], size=12, sub="per token MLP")
    s.arrow(x + 91, 148, x + 91, 168)
    x2 = x + 176
    s.arrow(x2, y + 32, x2 + 20, y + 32)
    s.box(x2 + 22, y, 120, 64, "Linear + softmax", C["orange_l"], C["orange"], size=12, sub="scores for every word")
    s.arrow(x2 + 142, y + 32, x2 + 160, y + 32)
    s.text(x2 + 164, y + 10, "next token", 12, 700, C["orange"], anchor="start")
    s.text(x2 + 164, y + 30, '"on"  62%', 12, 400, anchor="start", mono=True)
    s.text(x2 + 164, y + 47, '"down" 21%', 12, 400, anchor="start", mono=True)
    s.text(x2 + 164, y + 64, '"..."', 12, 400, anchor="start", mono=True)
    s.text(560, 300, "Topics 1 + 6 build the input (blue/teal). Topics 2-5 live inside the purple block. "
                     "Topics 7-10 decide how blocks are wired.", 12, 500, C["grey"])
    s.save("d01_big_picture.svg")


# ---------------------------------------------------------------- D2 embeddings
def embeddings():
    s = new(980, 360, "Topic 1 - Embeddings: from discrete IDs to points in space")
    s.text(130, 64, "Discrete: words are just IDs", 13, 700)
    words = [("cat", 0), ("mat", 1), ("on", 2), ("sat", 3), ("the", 4)]
    for i, (w, idx) in enumerate(words):
        s.box(40, 80 + i * 44, 80, 34, w, C["grey_l"], C["grey"])
        s.arrow(120, 97 + i * 44, 150, 97 + i * 44)
        s.box(152, 80 + i * 44, 60, 34, f"ID {idx}", "#fff", C["grey"], size=12)
    s.text(130, 316, "ID 0 vs ID 1 says nothing about meaning", 11, 400, C["red"])
    # table
    s.text(455, 64, "Embedding table W  (vocab x d_model)", 13, 700)
    vals = [[-0.31, -0.40, 0.80, -0.62], [1.93, 1.49, 0.90, -2.11], [-0.23, 0.04, -0.25, 0.86],
            [0.36, -0.69, -0.49, 0.24], [-0.31, -0.40, 0.80, -0.62]]
    vals[4] = [-0.76, 1.08, 0.80, 1.68]
    for i, row in enumerate(vals):
        hl = i == 0
        s.rect(330, 80 + i * 44, 250, 34, C["blue_l"] if hl else "#fff", C["blue"] if hl else "#cbd5e1", rx=4)
        s.text(322, 102 + i * 44, f"row {i}", 11, 500, C["grey"], anchor="end")
        for j, v in enumerate(row):
            s.text(362 + j * 60, 102 + i * 44, f"{v:+.2f}", 12, 600 if hl else 400, mono=True)
    s.arrow(212, 97, 290, 97, color=C["blue"], label="look up", ly=-10)
    s.text(455, 316, "These numbers start random and are LEARNED in training", 11, 400, C["blue"])
    # space
    s.text(800, 64, "Continuous: meaning = position", 13, 700)
    s.rect(660, 80, 300, 220, "#fff", "#cbd5e1", rx=6)
    pts = [("cat", 700, 130), ("dog", 735, 150), ("kitten", 690, 170),
           ("car", 880, 250), ("bus", 905, 225), ("truck", 860, 220),
           ("king", 860, 115), ("queen", 900, 135)]
    for w, x, y in pts:
        s.circle(x, y, 5, C["blue"], C["blue"])
        s.text(x + 8, y + 4, w, 12, 500, anchor="start")
    s.text(800, 316, "similar words end up close together (cosine similarity)", 11, 400, C["green"])
    s.save("d02_embeddings.svg")


# ---------------------------------------------------------------- D3 attention as lookup
def attention_lookup():
    s = new(980, 380, "Topic 2 - Attention = a soft, learned lookup")
    s.box(30, 160, 150, 70, "Query q = [1, 0]", C["orange_l"], C["orange"], sub='"I am looking for a noun"')
    keys = [("Key: cat", "Value: [10, 0]", "1.0", "0.42"), ("Key: sat", "Value: [0, 10]", "0.0", "0.16"),
            ("Key: mat", "Value: [5, 5]", "1.0", "0.42")]
    for i, (k, v, sc, w) in enumerate(keys):
        y = 70 + i * 100
        s.box(260, y, 130, 46, k, C["purple_l"], C["purple"], size=12)
        s.box(260, y + 48, 130, 30, v, C["green_l"], C["green"], size=11, weight=500)
        s.arrow(180, 195, 258, y + 23, color=C["orange"])
        s.box(450, y + 6, 90, 34, f"q.k = {sc}", "#fff", C["grey"], size=12, weight=500)
        s.arrow(390, y + 23, 448, y + 23)
        s.box(600, y + 6, 100, 34, f"weight {w}", C["yellow_l"], "#ca8a04", size=12)
        s.arrow(540, y + 23, 598, y + 23, label="softmax", ly=-8)
        s.arrow(700, y + 23, 768, 195)
    s.box(770, 160, 180, 70, "Output", C["orange_l"], C["orange"], sub="0.42*cat + 0.16*sat + 0.42*mat")
    s.text(860, 260, "= [6.33, 3.67]", 13, 700, mono=True)
    s.text(490, 365, "Score = how well the question matches each label. Softmax turns scores into "
                     "percentages. Output = blend of the values.", 12, 500, C["grey"])
    s.save("d03_attention_lookup.svg")


# ---------------------------------------------------------------- D4 self-attention matrices
def self_attention():
    s = new(1000, 340, "Topic 3 - Self-attention: Q, K and V all come from the SAME sentence")
    s.box(20, 135, 90, 70, "X", C["blue_l"], C["blue"], size=18, sub="T x d_model")
    for i, (n, col, colf) in enumerate([("Q", C["orange"], C["orange_l"]), ("K", C["purple"], C["purple_l"]),
                                        ("V", C["green"], C["green_l"])]):
        y = 60 + i * 90
        s.arrow(110, 170, 198, y + 27, label=f"x W_{n}", ly=-4)
        s.box(200, y, 70, 54, n, colf, col, size=18, sub="T x d_k")
    s.arrow(270, 87, 340, 140)
    s.arrow(270, 177, 340, 160)
    s.box(342, 120, 130, 60, "Q K^T", C["yellow_l"], "#ca8a04", size=16, sub="T x T scores")
    s.arrow(472, 150, 510, 150)
    s.box(512, 120, 100, 60, "/ sqrt(d_k)", "#fff", C["grey"], size=13, sub="keep calm")
    s.arrow(612, 150, 650, 150)
    s.box(652, 120, 100, 60, "softmax", "#fff", C["grey"], size=14, sub="rows sum to 1")
    s.arrow(752, 150, 800, 185)
    s.arrow(270, 267, 800, 205)
    s.box(802, 160, 70, 60, "x V", C["green_l"], C["green"], size=16)
    s.arrow(872, 190, 900, 190)
    s.box(902, 155, 80, 70, "Z", C["blue_l"], C["blue"], size=18, sub="T x d_k")
    s.text(500, 320, "Attention(Q, K, V) = softmax( Q K^T / sqrt(d_k) ) V      "
                     "- every word gets a new vector that mixes in the words it cares about",
           12.5, 600, C["grey"])
    s.save("d04_self_attention.svg")


# ---------------------------------------------------------------- D5 multi-head
def multi_head():
    s = new(1000, 380, "Topic 4 - Multi-head attention: several small attentions in parallel")
    s.box(20, 150, 100, 70, "X", C["blue_l"], C["blue"], size=18, sub="T x 512")
    names = ["head 1: grammar", "head 2: who-did-what", "head 3: nearby words", "head h: ..."]
    for i, n in enumerate(names):
        y = 50 + i * 75
        s.arrow(120, 185, 198, y + 28)
        s.rect(200, y, 330, 56, C["purple_l"], C["purple"])
        s.text(365, y + 22, n, 13, 700, C["purple"])
        s.text(365, y + 41, "own W_Q, W_K, W_V  ->  attention  ->  T x 64", 11, 400, C["grey"])
        s.arrow(530, y + 28, 598, 185)
    s.box(600, 150, 120, 70, "Concat", C["yellow_l"], "#ca8a04", size=15, sub="T x (h*64) = T x 512")
    s.arrow(720, 185, 768, 185)
    s.box(770, 150, 90, 70, "x W_O", C["grey_l"], C["grey"], size=15, sub="mix heads")
    s.arrow(860, 185, 888, 185)
    s.box(890, 150, 95, 70, "Out", C["blue_l"], C["blue"], size=18, sub="T x 512")
    s.text(500, 360, "Same total cost as one big head: d_model is SPLIT into h pieces "
                     "(512 = 8 heads x 64). Each head can learn a different relationship.", 12, 500, C["grey"])
    s.save("d05_multi_head.svg")


# ---------------------------------------------------------------- D6 masks
def masks():
    s = new(980, 380, "Topic 5 - Masked attention: no peeking at the future")
    toks = ["I", "love", "deep", "learning", "!"]
    s.text(170, 60, "Encoder / BERT: full attention", 13, 700)
    s.grid(90, 110, 5, 36, lambda i, j: True, toks)
    s.text(500, 60, "Decoder / GPT: causal mask", 13, 700)
    s.grid(420, 110, 5, 36, lambda i, j: j <= i, toks)
    s.text(810, 60, "Padding mask (2 real words)", 13, 700)
    s.grid(740, 110, 5, 36, lambda i, j: j <= 1, ["I", "ran", "PAD", "PAD", "PAD"])
    s.rect(90, 320, 18, 18, C["green"], C["green"], rx=3)
    s.text(114, 334, "allowed (mask = 1)", 12, 500, anchor="start")
    s.rect(270, 320, 18, 18, "#e2e8f0", "#e2e8f0", rx=3)
    s.text(294, 334, "blocked: score set to -inf, softmax gives 0", 12, 500, anchor="start")
    s.text(490, 366, "Rows = the word that is looking (query). Columns = the word being looked at (key).",
           12, 500, C["grey"])
    s.save("d06_masks.svg")


# ---------------------------------------------------------------- D7 positional / rope
def positions():
    s = new(980, 360, "Topic 6 - Positional encoding: telling the model WHERE each word is")
    s.text(160, 62, "1. Add a position vector", 13, 700)
    s.box(40, 85, 110, 50, "word vector", C["blue_l"], C["blue"], size=12, sub='"cat"')
    s.text(160, 117, "+", 22, 700)
    s.box(172, 85, 110, 50, "position vector", C["teal_l"], C["teal"], size=12, sub="slot #2")
    s.arrow(160, 140, 160, 175)
    s.box(95, 178, 130, 50, "input to block", "#fff", C["grey"], size=12, sub="meaning + place")
    s.text(160, 260, "Sinusoidal (2017): fixed sin/cos waves", 11.5, 400, anchor="middle")
    s.text(160, 278, "Learned (GPT-2, BERT): a trainable table", 11.5, 400, anchor="middle")
    # rope circle
    s.text(640, 62, "2. RoPE (Llama, Qwen, Mistral...): ROTATE q and k", 13, 700)
    cx, cy, r = 520, 200, 110
    s.circle(cx, cy, r, "#fff", "#cbd5e1")
    import math as m
    for k, (lab, col) in enumerate([("pos 0", C["blue"]), ("pos 1", C["teal"]), ("pos 2", C["purple"]),
                                    ("pos 3", C["orange"])]):
        ang = m.radians(10 + 32 * k)
        x2, y2 = cx + r * 0.9 * m.cos(ang), cy - r * 0.9 * m.sin(ang)
        s.arrow(cx, cy, round(x2, 1), round(y2, 1), color=col, sw=2.2)
        s.text(round(cx + r * 1.08 * m.cos(ang), 1), round(cy - r * 1.08 * m.sin(ang), 1) + 4, lab, 12, 600, col)
    s.text(cx, cy + r + 24, "same vector, rotated by (position x angle)", 11.5, 400, C["grey"])
    s.text(700, 140, "Why it works:", 12.5, 700, anchor="start")
    s.text(700, 162, "q at pos 30 . k at pos 28", 12, 400, anchor="start", mono=True)
    s.text(700, 180, "  == q at pos 3 . k at pos 1", 12, 400, anchor="start", mono=True)
    s.text(700, 204, "The score depends only on the", 12, 400, anchor="start")
    s.text(700, 222, "DISTANCE between words, not on", 12, 400, anchor="start")
    s.text(700, 240, "where the pair sits in the text.", 12, 400, anchor="start")
    s.text(700, 266, "You prove this in cell 5.3-D.", 12, 600, C["teal"], anchor="start")
    s.save("d07_positions.svg")


# ---------------------------------------------------------------- D8 transformer block
def block():
    s = new(620, 520, "One Transformer block (pre-norm, as used today)")
    x = 200
    s.box(x, 460, 220, 40, "x  (T x d_model)", C["blue_l"], C["blue"], size=13)
    s.arrow(x + 110, 460, x + 110, 420)
    s.box(x, 380, 220, 40, "LayerNorm", "#fff", C["grey"], size=13)
    s.arrow(x + 110, 380, x + 110, 350)
    s.box(x, 300, 220, 50, "Multi-head attention", C["purple_l"], C["purple"], size=13, sub="(masked if decoder)")
    s.arrow(x + 110, 300, x + 110, 278)
    s.circle(x + 110, 265, 13, "#fff")
    s.text(x + 110, 271, "+", 18, 700)
    s.path(f"M{x + 220},480 L{x + 300},480 L{x + 300},265 L{x + 125},265", color=C["red"], dash="6 4")
    s.text(x + 312, 380, "residual", 12, 600, C["red"], anchor="start")
    s.text(x + 312, 396, "(skip path)", 11, 400, C["red"], anchor="start")
    s.arrow(x + 110, 252, x + 110, 222)
    s.box(x, 182, 220, 40, "LayerNorm", "#fff", C["grey"], size=13)
    s.arrow(x + 110, 182, x + 110, 152)
    s.box(x, 102, 220, 50, "Feed-forward (MLP)", C["green_l"], C["green"], size=13, sub="d -> 4d -> d, per token")
    s.arrow(x + 110, 102, x + 110, 80)
    s.circle(x + 110, 67, 13, "#fff")
    s.text(x + 110, 73, "+", 18, 700)
    s.path(f"M{x + 110},240 L{x - 70},240 L{x - 70},67 L{x + 95},67", color=C["red"], dash="6 4")
    s.arrow(x + 110, 54, x + 110, 40)
    s.text(x + 130, 46, "out (same shape as x)", 12, 600, anchor="start")
    s.text(60, 330, "attention = words", 11.5, 600, C["purple"])
    s.text(60, 346, "talk to each other", 11.5, 400, C["purple"])
    s.text(60, 125, "MLP = each word", 11.5, 600, C["green"])
    s.text(60, 141, "thinks alone", 11.5, 400, C["green"])
    s.save("d08_block.svg")


# ---------------------------------------------------------------- D9 encoder-decoder
def encoder_decoder():
    s = new(980, 560, "Topic 7 - Encoder-decoder Transformer (original 2017 design, T5, BART, Whisper)")
    # encoder
    s.rect(60, 120, 300, 300, C["blue_l"], C["blue"], rx=12)
    s.text(210, 145, "ENCODER  (N x)", 14, 700, C["blue"])
    s.box(90, 170, 240, 60, "Multi-head self-attention", "#fff", C["purple"], size=13, sub="full: every word sees every word")
    s.box(90, 260, 240, 50, "Feed-forward", "#fff", C["green"], size=13)
    s.arrow(210, 260, 210, 232)
    s.text(210, 345, "Add & Norm after each", 11, 400, C["grey"])
    s.text(210, 395, "reads the WHOLE input at once", 12, 600, C["blue"])
    s.box(110, 460, 200, 44, '"I love AI"', C["grey_l"], C["grey"], size=13, sub="embedding + position")
    s.arrow(210, 460, 210, 422)
    # decoder
    s.rect(560, 80, 340, 380, C["orange_l"], C["orange"], rx=12)
    s.text(730, 105, "DECODER  (N x)", 14, 700, C["orange"])
    s.box(590, 330, 280, 56, "MASKED multi-head self-attention", "#fff", C["purple"], size=12.5, sub="only looks at words already written")
    s.box(590, 225, 280, 66, "CROSS-attention", C["yellow_l"], "#ca8a04", size=14, sub="Q from decoder, K and V from encoder")
    s.box(590, 130, 280, 50, "Feed-forward", "#fff", C["green"], size=13)
    s.arrow(730, 330, 730, 293)
    s.arrow(730, 225, 730, 182)
    s.box(630, 490, 200, 44, '"<s> J\'aime l\'"', C["grey_l"], C["grey"], size=13, sub="what it wrote so far")
    s.arrow(730, 490, 730, 388)
    # memory arrow
    s.path("M360,200 C450,200 480,258 588,258", color=C["red"], sw=2.5)
    s.text(470, 190, "encoder output", 12, 700, C["red"])
    s.text(470, 206, '= "memory" (K, V)', 12, 500, C["red"])
    s.arrow(730, 80, 730, 50)
    s.text(730, 42, 'Linear + softmax  ->  next word: "IA"', 13, 700, C["orange"])
    s.save("d09_encoder_decoder.svg")


# ---------------------------------------------------------------- D10 three families
def families():
    s = new(1000, 470, "Topics 8, 7, 9 - The three Transformer families")
    cols = [
        ("Encoder-only", "BERT, RoBERTa, ModernBERT", C["blue"], C["blue_l"], lambda i, j: True,
         ["understand text", "classification, NER,", "search embeddings, RAG"], "fill in the [MASK]"),
        ("Encoder-decoder", "T5, BART, Whisper, NLLB", C["orange"], C["orange_l"], None,
         ["text in -> text out", "translation, summaries,", "speech-to-text"], "rewrite input -> output"),
        ("Decoder-only", "GPT, Llama, Qwen, Claude-style", C["purple"], C["purple_l"], lambda i, j: j <= i,
         ["generate text", "chat, code, agents,", "almost every LLM today"], "predict the next token"),
    ]
    for k, (name, ex, col, colf, rule, uses, obj) in enumerate(cols):
        x = 30 + k * 325
        s.rect(x, 50, 300, 400, colf, col, rx=12)
        s.text(x + 150, 80, name, 16, 700, col)
        s.text(x + 150, 100, ex, 11.5, 500, C["grey"])
        if rule:
            s.grid(x + 90, 135, 5, 26, rule, None, on=col)
            s.text(x + 150, 290, "attention pattern", 11, 500, C["grey"])
        else:
            s.grid(x + 30, 135, 5, 22, lambda i, j: True, None, on=C["blue"])
            s.grid(x + 165, 135, 5, 22, lambda i, j: j <= i, None, on=C["purple"])
            s.text(x + 85, 265, "encoder: full", 10.5, 500, C["grey"])
            s.text(x + 220, 265, "decoder: causal", 10.5, 500, C["grey"])
            s.text(x + 150, 290, "+ cross-attention between them", 11, 600, "#ca8a04")
        s.text(x + 150, 320, f"Training goal: {obj}", 12, 600)
        for u, line in enumerate(uses):
            s.text(x + 150, 350 + u * 20, line, 12.5, 500)
    s.save("d10_families.svg")


# ---------------------------------------------------------------- D11 cross attention
def cross_attention():
    s = new(980, 400, "Topic 10 - Cross-attention: the decoder asks, the encoder answers")
    src = ["3", "1", "4", "1", "5"]
    s.text(150, 60, "Encoder output (memory)", 13, 700, C["blue"])
    for i, t in enumerate(src):
        s.box(40 + i * 46, 75, 40, 40, t, C["blue_l"], C["blue"], size=14)
    s.arrow(150, 118, 150, 160, color=C["purple"], label="x W_K", lx=24, ly=0)
    s.arrow(110, 118, 60, 300, color=C["green"], label="x W_V", lx=-34, ly=0)
    s.box(80, 165, 140, 40, "K  (5 keys)", C["purple_l"], C["purple"], size=13)
    s.box(20, 305, 140, 40, "V  (5 values)", C["green_l"], C["green"], size=13)
    s.text(830, 60, "Decoder states (what it wrote)", 13, 700, C["orange"])
    dec = ["<s>", "5", "1", "4"]
    for i, t in enumerate(dec):
        s.box(740 + i * 50, 75, 44, 40, t, C["orange_l"], C["orange"], size=13)
    s.arrow(830, 118, 830, 160, color=C["orange"], label="x W_Q", lx=24, ly=0)
    s.box(760, 165, 140, 40, "Q  (4 queries)", C["orange_l"], C["orange"], size=13)
    s.arrow(760, 185, 560, 230)
    s.arrow(220, 185, 400, 230)
    s.text(490, 225, "Q K^T  ->  4 x 5 score grid", 13, 700)
    vals = [[0.05, 0.05, 0.05, 0.05, 0.8], [0.05, 0.05, 0.05, 0.8, 0.05], [0.05, 0.05, 0.8, 0.05, 0.05],
            [0.05, 0.8, 0.05, 0.05, 0.05]]
    for i in range(4):
        for j in range(5):
            op = 0.15 + 0.85 * vals[i][j] / 0.8
            s.parts.append(f'<rect x="{420 + j * 32}" y="{240 + i * 30}" width="30" height="28" rx="3" '
                           f'fill="{C["purple"]}" fill-opacity="{op:.2f}"/>')
    s.text(490, 375, "each decoder step points at the source digit it needs", 11.5, 500, C["grey"])
    s.arrow(160, 325, 410, 300, color=C["green"])
    s.arrow(590, 300, 740, 300)
    s.box(742, 275, 210, 50, "weighted mix of V", C["green_l"], C["green"], size=13, sub="-> next output digit")
    s.save("d11_cross_attention.svg")


# ---------------------------------------------------------------- D12 kv cache
def kv_cache():
    s = new(980, 340, "Elite - KV cache: never recompute the past")
    s.text(245, 60, "Without cache: step 4 recomputes K,V for ALL 4 tokens", 13, 700, C["red"])
    for step in range(4):
        y = 80 + step * 52
        s.text(40, y + 26, f"step {step + 1}", 12, 600, anchor="start")
        for t in range(step + 1):
            s.rect(110 + t * 70, y + 6, 60, 32, C["red_l"], C["red"], rx=5)
            s.text(140 + t * 70, y + 27, f"K,V {t + 1}", 11, 600)
    s.text(245, 310, "total work grows like n^2", 12, 500, C["red"])
    s.text(735, 60, "With cache: compute ONE new K,V, reuse the rest", 13, 700, C["green"])
    for step in range(4):
        y = 80 + step * 52
        s.text(530, y + 26, f"step {step + 1}", 12, 600, anchor="start")
        for t in range(step + 1):
            new_one = t == step
            s.rect(600 + t * 70, y + 6, 60, 32, C["green_l"] if new_one else C["grey_l"],
                   C["green"] if new_one else "#94a3b8", rx=5)
            s.text(630 + t * 70, y + 27, f"K,V {t + 1}", 11, 600 if new_one else 400)
    s.text(735, 310, "grey = read from memory, green = new work", 12, 500, C["green"])
    s.save("d12_kv_cache.svg")


# ---------------------------------------------------------------- D13 attention variants
def variants():
    s = new(1000, 330, "Elite - How modern models shrink the KV cache")
    blocks = [("MHA", "1 K/V head per query head", 8, 8), ("GQA (Llama 3, Qwen3)", "query heads share K/V in groups", 8, 2),
              ("MQA", "all heads share 1 K/V", 8, 1)]
    for b, (name, sub, nq, nkv) in enumerate(blocks):
        x0 = 20 + b * 250
        s.text(x0 + 110, 60, name, 14, 700)
        s.text(x0 + 110, 78, sub, 11, 400, C["grey"])
        for q in range(nq):
            s.rect(x0 + 10 + q * 26, 100, 20, 30, C["orange_l"], C["orange"], rx=3)
        group = nq // nkv
        kv_w = (nq * 26) / nkv
        for k in range(nkv):
            kx = x0 + 10 + k * kv_w + kv_w / 2 - 10
            s.rect(kx, 210, 20, 30, C["purple_l"], C["purple"], rx=3)
            for q in range(k * group, (k + 1) * group):
                s.parts.append(f'<line x1="{x0 + 20 + q * 26}" y1="130" x2="{kx + 10}" y2="210" '
                               f'stroke="#94a3b8" stroke-width="1"/>')
        s.text(x0 + 110, 270, f"{nkv} K/V head{'s' if nkv > 1 else ''} stored", 12, 600, C["purple"])
    x0 = 770
    s.text(x0 + 110, 60, "MLA (DeepSeek-V3)", 14, 700)
    s.text(x0 + 110, 78, "store one small compressed latent", 11, 400, C["grey"])
    for q in range(8):
        s.rect(x0 + 10 + q * 26, 100, 20, 30, C["orange_l"], C["orange"], rx=3)
    s.rect(x0 + 60, 200, 100, 26, C["teal_l"], C["teal"], rx=4)
    s.text(x0 + 110, 218, "latent c", 12, 600, C["teal"])
    s.arrow(x0 + 110, 198, x0 + 110, 134, color=C["teal"], label="expand to K,V", lx=48, ly=4)
    s.text(x0 + 110, 270, "tiny cache, near-MHA quality", 12, 600, C["teal"])
    s.text(500, 312, "orange = query heads, purple = key/value heads you must keep in GPU memory "
                     "for every past token", 12, 500, C["grey"])
    s.save("d13_attention_variants.svg")


if __name__ == "__main__":
    for fn in [env_map, big_picture, embeddings, attention_lookup, self_attention, multi_head, masks,
               positions, block, encoder_decoder, families, cross_attention, kv_cache, variants]:
        fn()
    print("diagrams written to", HERE)
