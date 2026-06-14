"""
Post #65 designed social asset — AI-Agent-Conversation-Quality-Scorer, Phase 4.
Style #4 (architecture / methodology decision): "the model was never the bottleneck."

Carries the post's first-paragraph claim: tuning moved the score by exactly 0.0000.
A "ledger" of every model-side lever, each landing flat, plus the takeaway that the
13 engineered features already carried all the signal.

Numbers on the graphic (0.0000, 0.9808, "made it worse", 13 features) all match the
post body and the Phase-4 report exactly. No matplotlib — pure PIL pull-quote graphic.
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
draw_tracked(ebx, 92, "  ·  WHAT MOVED THE SCORE", eb_font, DIMSLATE, track=2)

# ---- headline ----
hl_font = f(FB, 80)
y = 150
d.text((M, y), "Tuning moved the score", font=hl_font, fill=OFFWHITE)
y += 92
# second line: "by exactly " + accent "0.0000."
seg1 = "by exactly "
d.text((M, y), seg1, font=hl_font, fill=OFFWHITE)
d.text((M + tw(seg1, hl_font), y), "0.0000", font=hl_font, fill=AMBER)
d.text((M + tw(seg1 + "0.0000", hl_font), y), ".", font=hl_font, fill=OFFWHITE)

# ---- context line ----
y += 120
ctx_font = f(FR, 26)
d.text((M, y), "Every model-side lever on a strong hallucination detector.", font=ctx_font, fill=SLATE)
y += 36
d.text((M, y), "What each one actually changed:", font=ctx_font, fill=SLATE)

# ---- ledger of levers ----
rows = [
    ("5 min of Bayesian tuning, 9 knobs", "+0.0000", ROSE),
    ("XGBoost = LightGBM = CatBoost", "0.9808", SLATE),
    ("Calibration, the textbook fix", "made it worse", ROSE),
]
lab_font = f(FB, 31)
val_font = f(FBL, 38)
val_font_sm = f(FBL, 30)

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
    vf = val_font if len(val) <= 8 else val_font_sm
    vw = tw(val, vf)
    vh = (vf.getbbox(val)[3] - vf.getbbox(val)[1])
    d.text((x1 - 34 - vw, y0 + (row_h - vh) / 2 - vf.getbbox(val)[1]),
           val, font=vf, fill=vcol)
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
d.text((strip_x0 + 34, strip_y0 + 28), "THE MODEL WAS NEVER THE BOTTLENECK", font=take_font, fill=EMERALD)
d.text((strip_x0 + 34, strip_y0 + 74),
       "13 engineered features already carried all the signal.", font=sub_font, fill=OFFWHITE)

# ---- footer ----
foot_font = f(FI, 20)
foot = "HaluEval-QA  ·  20,000 balanced QA samples  ·  length-matched macro-F1  ·  grouped CV"
d.text((M, H - 70), foot, font=foot_font, fill=DIMSLATE)

img.save("/Users/anthonyrodrigues/Desktop/YC-Portfolio-Projects/AI-Agent-Conversation-Quality-Scorer/results/posts/post65_social.png")
print("saved post65_social.png")
