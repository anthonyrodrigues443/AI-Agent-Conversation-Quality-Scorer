"""
Designed social asset for Post #61 — AI-Agent-Conversation-Quality-Scorer, Phase 1.
Claim carried: a model that only counts characters scored 94.4% on HaluEval
(where ChatGPT scores 62.6%); a length-matched control collapses it to 61.5%,
while grounding-overlap survives at 91.9%.

PIL-only (no matplotlib). 1080x1080, dark navy, typography-driven stat card.
"""
from PIL import Image, ImageDraw, ImageFont

W = H = 1080
BG      = (15, 23, 42)      # #0F172A slate-900
OFFW    = (250, 250, 249)   # #FAFAF9 off-white
SLATE   = (148, 163, 184)   # #94A3B8 muted label
DIM     = (203, 213, 225)   # #CBD5E1 dim number (the collapse)
AMBER   = (251, 191, 36)    # #FBBF24 the inflated "win"
GREEN   = (52, 211, 153)    # #34D399 the real signal
SKY     = (56, 189, 248)    # #38BDF8 eyebrow
RULE    = (30, 41, 59)      # #1E293B hairline
PANEL_G = (16, 42, 34)      # dark-green survivor panel
FOOT    = (100, 116, 139)   # #64748B footer

AB = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
HV = "/System/Library/Fonts/Helvetica.ttc"

def ab(sz): return ImageFont.truetype(AB, sz)
def hv(sz): return ImageFont.truetype(HV, sz)

img = Image.new("RGB", (W, H), BG)
d = ImageDraw.Draw(img)

LX, RX = 90, 990  # content margins

def tracked(draw, x, y, text, font, fill, tracking):
    for ch in text:
        draw.text((x, y), ch, font=font, fill=fill, anchor="lm")
        x += draw.textlength(ch, font=font) + tracking
    return x

# --- Eyebrow ---
tracked(d, LX, 104, "HALLUCINATION DETECTION   ·   HALUEVAL-QA", hv(23), SKY, 4)

# --- Headline (the post's first-paragraph claim, compressed) ---
hf = ab(78)
d.text((LX, 168), "It ‘beat’ ChatGPT by", font=hf, fill=OFFW, anchor="lt")
d.text((LX, 256), "counting characters.", font=hf, fill=OFFW, anchor="lt")

# hairline rule
d.line([(LX, 388), (RX, 388)], fill=RULE, width=2)

# --- Three-beat arc: label (left) + number (right) ---
labf = hv(31)
numf = ab(74)
subf = hv(24)

rows = [
    (470, "ChatGPT zero-shot, from the paper", "62.6%", SLATE, None),
    (588, "A classifier that only counts characters", "94.4%", AMBER, "looks like a huge win"),
    (706, "Same model, lengths matched", "61.5%", DIM, "≈ coin flip"),
]
for cy, label, num, col, sub in rows:
    d.text((LX, cy), label, font=labf, fill=OFFW if col is AMBER else SLATE, anchor="lm")
    d.text((RX, cy), num, font=numf, fill=col, anchor="rm")
    if sub:
        scol = AMBER if col is AMBER else (251, 113, 133)  # rose for the collapse note
        d.text((RX, cy + 46), sub, font=subf, fill=scol, anchor="rm")

# collapse connector between row 2 and row 3
d.text((LX, 647), "length-matched control", font=hv(22), fill=(251, 113, 133), anchor="lm")
d.text((LX + 250, 647), "↓", font=ab(30), fill=(251, 113, 133), anchor="lm")

# --- Survivor strip ---
d.rounded_rectangle([LX, 792, RX, 902], radius=22, fill=PANEL_G)
d.rectangle([LX, 792, LX + 8, 902], fill=GREEN)  # accent bar
tracked(d, LX + 40, 833, "THE ONE SIGNAL THAT SURVIVED", hv(21), GREEN, 3)
d.text((LX + 40, 868), "Grounding overlap with the source", font=hv(27), fill=OFFW, anchor="lm")
d.text((RX - 34, 847), "91.9%", font=ab(58), fill=GREEN, anchor="rm")

# --- Footer ---
d.text((LX, 992), "HaluEval-QA  ·  20,000 balanced QA samples  ·  length-matched control  ·  macro-F1",
       font=ImageFont.truetype(HV, 22), fill=FOOT, anchor="lm")

out = "/Users/anthonyrodrigues/Desktop/YC-Portfolio-Projects/AI-Agent-Conversation-Quality-Scorer/results/posts/post61_social.png"
img.save(out, "PNG")
print("saved", out)
