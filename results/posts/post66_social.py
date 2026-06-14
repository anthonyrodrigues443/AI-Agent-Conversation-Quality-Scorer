"""
Post #66 designed social asset — AI-Agent-Conversation-Quality-Scorer, Phase 5.
Style #8 (model comparison): a 13-feature tree vs frontier LLMs, the mirror image.

Carries the post's first-paragraph claim: a tiny tree beat Opus 4.8 and GPT-5.5 at
catching AI hallucinations on the average, then was totally blind (0/10) to the one
class only frontier reasoning catches (GPT-5.5 10/10).

Two stacked leaderboards. The tree sits at the TOP of "THE AVERAGE" (emerald winner)
and the BOTTOM of "THE HARD CASE" (rose loser) — the visual mirror is the point.

Every number on the graphic (1.000, 0.90, 0.90, 0.82, 10/10, 9/10, 5/10, 0/10) matches
the post body exactly. No new metrics introduced visually. Pure PIL, no matplotlib.
"""
from PIL import Image, ImageDraw, ImageFont

W = H = 1080
M = 80  # outer margin

# palette
BG        = "#0F172A"  # slate-900
CARD      = "#1E293B"  # slate-800
ROWALT    = "#172033"  # slightly lighter row band
OFFWHITE  = "#F8FAFC"
SLATE     = "#94A3B8"
DIMSLATE  = "#64748B"
SKY       = "#38BDF8"
AMBER     = "#FBBF24"
ROSE      = "#FB7185"
EMERALD   = "#34D399"
ROSEBG    = "#2A1620"
EMBG      = "#0B2A22"

FB  = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
FBL = "/System/Library/Fonts/Supplemental/Arial Black.ttf"
FR  = "/System/Library/Fonts/Supplemental/Arial.ttf"
FI  = "/System/Library/Fonts/Supplemental/Arial Italic.ttf"

img = Image.new("RGB", (W, H), BG)
d = ImageDraw.Draw(img)


def f(path, size):
    return ImageFont.truetype(path, size)


def tw(text, font):
    return d.textlength(text, font=font)


def draw_tracked(x, y, text, font, fill, track=0):
    cx = x
    for ch in text:
        d.text((cx, y), ch, font=font, fill=fill)
        cx += d.textlength(ch, font=font) + track
    return cx


def fit_font(path, text, max_w, start, floor=22):
    """largest size <= start whose text width <= max_w."""
    s = start
    while s > floor:
        fo = f(path, s)
        if tw(text, fo) <= max_w:
            return fo
        s -= 1
    return f(path, floor)


def vcenter_y(text, font, box_top, box_h):
    bb = font.getbbox(text)
    return box_top + (box_h - (bb[3] - bb[1])) / 2 - bb[1]


usable = W - 2 * M

# ---- eyebrow ----
eb_font = f(FB, 21)
x = draw_tracked(M, 86, "HALLUCINATION DETECTION", eb_font, SKY, track=2)
draw_tracked(x, 86, "   ·   TINY TREE vs FRONTIER LLMs", eb_font, DIMSLATE, track=2)

# ---- headline (2 lines, shrink-to-fit) ----
hl1 = "A tiny tree beat Opus 4.8 and GPT-5.5."
hl2 = "Then came the one fake it can't see."
hf1 = fit_font(FB, hl1, usable, 60)
hf2 = fit_font(FB, hl2, usable, 60)
y = 138
d.text((M, y), hl1, font=hf1, fill=OFFWHITE)
y += 70
d.text((M, y), hl2, font=hf2, fill=OFFWHITE)

# ---- context line ----
y += 78
ctx_font = f(FR, 25)
d.text((M, y), "Catching AI hallucinations, zero-shot. The same models, two cuts of the data:",
       font=ctx_font, fill=SLATE)

# ---- panel renderer ----
hdr_font   = f(FB, 22)
hdr_dim    = f(FB, 22)
name_font  = f(FB, 30)
val_font   = f(FBL, 34)


def panel(top, header_accent, header_dim, rows, height):
    x0, y0, x1, y1 = M, top, M + usable, top + height
    d.rounded_rectangle([x0, y0, x1, y1], radius=20, fill=CARD)
    # header band
    pad = 30
    hy = y0 + 26
    hx = draw_tracked(x0 + pad, hy, header_accent, hdr_font, SKY, track=1)
    draw_tracked(hx, hy, header_dim, hdr_dim, DIMSLATE, track=1)
    # divider
    d.line([x0 + pad, y0 + 62, x1 - pad, y0 + 62], fill="#2C3A52", width=2)
    # rows
    n = len(rows)
    rows_top = y0 + 76
    rows_area = (y1 - 18) - rows_top
    rh = rows_area / n
    for i, (name, val, vcol, accent) in enumerate(rows):
        ry = rows_top + i * rh
        # accent left bar for the highlighted row
        if accent:
            bar_col = EMERALD if accent == "win" else ROSE
            band_col = EMBG if accent == "win" else ROSEBG
            d.rounded_rectangle([x0 + 14, ry + 6, x1 - 14, ry + rh - 6], radius=12, fill=band_col)
            d.rounded_rectangle([x0 + 14, ry + 6, x0 + 21, ry + rh - 6], radius=4, fill=bar_col)
        ncol = OFFWHITE if accent else SLATE
        d.text((x0 + pad + 4, vcenter_y(name, name_font, ry, rh)), name, font=name_font, fill=ncol)
        vw = tw(val, val_font)
        d.text((x1 - pad - vw, vcenter_y(val, val_font, ry, rh)), val, font=val_font, fill=vcol)


# Panel 1 — THE AVERAGE
p1_rows = [
    ("13-feature tree (CPU)", "1.000", EMERALD, "win"),
    ("Claude Haiku 4.5",      "0.90",  SLATE,   None),
    ("GPT-5.5",               "0.90",  SLATE,   None),
    ("Claude Opus 4.8",       "0.82",  SLATE,   None),
]
p1_top = y + 56
p_h = 280
panel(p1_top, "THE AVERAGE", "   ·   macro-F1, representative set", p1_rows, p_h)

# Panel 2 — THE HARD CASE
p2_rows = [
    ("GPT-5.5",               "10/10", EMERALD, "win"),
    ("Claude Haiku 4.5",      "9/10",  SLATE,   None),
    ("Claude Opus 4.8",       "5/10",  SLATE,   None),
    ("13-feature tree",       "0/10",  ROSE,    "lose"),
]
p2_top = p1_top + p_h + 22
panel(p2_top, "THE HARD CASE", "   ·   verbatim quote, wrong answer · caught / 10", p2_rows, p_h)

# ---- footer ----
foot = "HaluEval-QA  ·  zero-shot frontier LLMs vs a 13-feature gradient-boosted tree  ·  tree ~0.03 ms, thousands of times cheaper"
ff = fit_font(FI, foot, usable, 20, floor=15)
d.text((M, H - 58), foot, font=ff, fill=DIMSLATE)

out = "/Users/anthonyrodrigues/Desktop/YC-Portfolio-Projects/AI-Agent-Conversation-Quality-Scorer/results/posts/post66_social.png"
img.save(out)
print("saved", out)
