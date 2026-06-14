"""
Post #67 designed social asset — AI-Agent-Conversation-Quality-Scorer, Phase 5.
Style #2 (metric / experiment comparison): the leave-one-out feature ablation.

Carries the post's first-paragraph claim: I engineered 13 features, removed them
one by one, and only one mattered. A "removal ledger" showing what each drop cost
the length-matched macro-F1, plus the takeaway that a 13-feature champion is really
a 1-feature model in disguise.

Numbers on the graphic (0.9808, -0.0742, 0.0000) all match the post body and the
Phase-5 report (Experiment 5.2) exactly. No matplotlib — pure PIL pull-quote graphic.
"""
from PIL import Image, ImageDraw, ImageFont

W = H = 1080
M = 80  # outer margin

# palette
BG        = "#0F172A"  # slate-900
CARD      = "#1E293B"  # slate-800
OFFWHITE  = "#F8FAFC"
SLATE     = "#94A3B8"
DIMSLATE  = "#64748B"
SKY       = "#38BDF8"
AMBER     = "#FBBF24"
ROSE      = "#FB7185"
EMERALD   = "#34D399"

FB = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
FBL = "/System/Library/Fonts/Supplemental/Arial Black.ttf"
FR = "/System/Library/Fonts/Supplemental/Arial.ttf"
FI = "/System/Library/Fonts/Supplemental/Arial Italic.ttf"

def f(path, size):
    return ImageFont.truetype(path, size)

img = Image.new("RGB", (W, H), BG)
d = ImageDraw.Draw(img)

def tw(text, font):
    return d.textlength(text, font=font)

def draw_tracked(x, y, text, font, fill, track=0):
    """draw text with manual letter-spacing (tracking) in px."""
    cx = x
    for ch in text:
        d.text((cx, y), ch, font=font, fill=fill)
        cx += d.textlength(ch, font=font) + track
    return cx

# ---- eyebrow ----
eb_font = f(FB, 21)
draw_tracked(M, 92, "HALLUCINATION DETECTION", eb_font, SKY, track=2)
ebx = M + sum(d.textlength(c, font=eb_font) + 2 for c in "HALLUCINATION DETECTION")
draw_tracked(ebx, 92, "  ·  LEAVE-ONE-OUT ABLATION", eb_font, DIMSLATE, track=2)

# ---- headline ----
hl_font = f(FB, 80)
y = 150
d.text((M, y), "13 features.", font=hl_font, fill=OFFWHITE)
y += 92
# second line: "Only " + accent "one" + " mattered."
seg1 = "Only "
d.text((M, y), seg1, font=hl_font, fill=OFFWHITE)
d.text((M + tw(seg1, hl_font), y), "one", font=hl_font, fill=AMBER)
d.text((M + tw(seg1 + "one", hl_font), y), " mattered.", font=hl_font, fill=OFFWHITE)

# ---- context line ----
y += 120
ctx_font = f(FR, 26)
d.text((M, y), "Full 13-feature model: 0.9808 macro-F1.", font=ctx_font, fill=SLATE)
y += 36
d.text((M, y), "Then I dropped each feature and retrained. What it cost:", font=ctx_font, fill=SLATE)

# ---- removal ledger ----
rows = [
    ("Drop lcs_char_ratio", "−0.0742", ROSE),
    ("Drop is_substr (was #1 by split)", "0.0000", SLATE),
    ("Drop any of the other 11", "0.0000", SLATE),
]
lab_font = f(FB, 31)
val_font = f(FBL, 38)

y += 70
row_h = 104
gap = 18
card_w = W - 2 * M
for label, val, vcol in rows:
    x0, y0 = M, y
    x1, y1 = M + card_w, y + row_h
    d.rounded_rectangle([x0, y0, x1, y1], radius=18, fill=CARD)
    # label, vertically centered
    lab_h = (lab_font.getbbox(label)[3] - lab_font.getbbox(label)[1])
    d.text((x0 + 34, y0 + (row_h - lab_h) / 2 - lab_font.getbbox(label)[1]),
           label, font=lab_font, fill=OFFWHITE)
    # value, right-aligned
    vw = tw(val, val_font)
    vh = (val_font.getbbox(val)[3] - val_font.getbbox(val)[1])
    d.text((x1 - 34 - vw, y0 + (row_h - vh) / 2 - val_font.getbbox(val)[1]),
           val, font=val_font, fill=vcol)
    y += row_h + gap

# ---- takeaway strip ----
y += 22
strip_x0, strip_y0 = M, y
strip_x1, strip_y1 = M + card_w, y + 132
d.rounded_rectangle([strip_x0, strip_y0, strip_x1, strip_y1], radius=18, fill="#0B2A22")
# emerald left accent bar
d.rounded_rectangle([strip_x0, strip_y0, strip_x0 + 8, strip_y1], radius=4, fill=EMERALD)
take_font = f(FB, 30)
sub_font = f(FR, 25)
d.text((strip_x0 + 34, strip_y0 + 28), "A 13-FEATURE MODEL, REALLY A 1-FEATURE MODEL", font=take_font, fill=EMERALD)
d.text((strip_x0 + 34, strip_y0 + 74),
       "Split importance shows what it used. Removal shows what it would lose.", font=sub_font, fill=OFFWHITE)

# ---- footer ----
foot_font = f(FI, 20)
foot = "HaluEval-QA  ·  length-matched macro-F1  ·  leave-one-out, retrained from scratch"
d.text((M, H - 70), foot, font=foot_font, fill=DIMSLATE)

img.save("/Users/anthonyrodrigues/Desktop/YC-Portfolio-Projects/AI-Agent-Conversation-Quality-Scorer/results/posts/post67_social.png")
print("saved post67_social.png")
