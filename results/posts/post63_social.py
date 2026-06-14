"""
Designed social asset for Post #63 — AI-Agent-Conversation-Quality-Scorer, Phase 2.
Claim carried: a 184M zero-shot NLI "meaning" model, asked to separate grounded
answers from hallucinated ones, scored the hallucinations as MORE entailed by the
source (0.26) than the true answers (0.21) — and lost the length-matched macro-F1
scoreboard to a one-line word-overlap rule (0.69 vs 0.92).

PIL-only (no matplotlib). 1080x1080, dark navy, typography-driven stat card.
"""
from PIL import Image, ImageDraw, ImageFont

W = H = 1080
BG      = (15, 23, 42)      # #0F172A slate-900
OFFW    = (250, 250, 249)   # #FAFAF9 off-white
SLATE   = (148, 163, 184)   # #94A3B8 muted label
DIM     = (203, 213, 225)   # #CBD5E1 dim number
ROSE    = (251, 113, 133)   # #FB7185 the wrong-higher signal
AMBER   = (251, 191, 36)    # #FBBF24
GREEN   = (52, 211, 153)    # #34D399 the winner
SKY     = (56, 189, 248)    # #38BDF8 eyebrow
RULE    = (30, 41, 59)      # #1E293B hairline
PANEL   = (24, 35, 56)      # slate-800-ish card fill
PANEL_R = (40, 24, 33)      # dark-rose card fill (the inverted one)
PANEL_S = (17, 28, 30)      # dark-slate scoreboard strip
FOOT    = (100, 116, 139)   # #64748B footer

AB = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
HV = "/System/Library/Fonts/Helvetica.ttc"

def ab(sz): return ImageFont.truetype(AB, sz)
def hv(sz): return ImageFont.truetype(HV, sz)

img = Image.new("RGB", (W, H), BG)
d = ImageDraw.Draw(img)

LX, RX = 90, 990

def tracked(draw, x, y, text, font, fill, tracking):
    for ch in text:
        draw.text((x, y), ch, font=font, fill=fill, anchor="lm")
        x += draw.textlength(ch, font=font) + tracking
    return x

# --- Eyebrow ---
tracked(d, LX, 96, "HALLUCINATION DETECTION   ·   ZERO-SHOT NLI vs A ONE-LINE RULE", hv(21), SKY, 3)

# --- Headline (the post's opening claim) ---
hf = ab(70)
d.text((LX, 150), "It scored the fakes as more", font=hf, fill=OFFW, anchor="lt")
d.text((LX, 232), "grounded than the truth.", font=hf, fill=OFFW, anchor="lt")

# context line under headline
d.text((LX, 326), "Asked to flag hallucinated answers, a 184M “meaning” model (NLI)",
       font=hv(25), fill=SLATE, anchor="lt")
d.text((LX, 360), "got the signal backwards. Here is what it believed:",
       font=hv(25), fill=SLATE, anchor="lt")

# --- Inversion hero: two panels ---
PT, PB = 420, 672
midgap = 24
lp_x0, lp_x1 = LX, (LX + RX) // 2 - midgap // 2
rp_x0, rp_x1 = (LX + RX) // 2 + midgap // 2, RX

d.rounded_rectangle([lp_x0, PT, lp_x1, PB], radius=22, fill=PANEL)
d.rounded_rectangle([rp_x0, PT, rp_x1, PB], radius=22, fill=PANEL_R)
d.rectangle([rp_x0, PT, rp_x0 + 8, PB], fill=ROSE)  # accent bar on the inverted panel

def panel_block(cx, label, lcol, num, ncol, sub, subcol):
    d.text((cx, PT + 40), label, font=hv(23), fill=lcol, anchor="mm")
    d.text((cx, PT + 128), num, font=ab(108), fill=ncol, anchor="mm")
    d.text((cx, PT + 210), sub, font=hv(22), fill=subcol, anchor="mm")

panel_block((lp_x0 + lp_x1) // 2, "GROUNDED (TRUE) ANSWERS", SLATE,
            "0.21", DIM, "avg P(entailed)", SLATE)
panel_block((rp_x0 + rp_x1) // 2, "HALLUCINATIONS", OFFW,
            "0.26", ROSE, "avg P(entailed)  ·  HIGHER", ROSE)

# caption under the panels
d.text((LX, PB + 32), "The model gave the made-up answers MORE entailment than the true ones.",
       font=hv(25), fill=OFFW, anchor="lt")

# --- Scoreboard strip (two-row mini-leaderboard) ---
ST, SB = 750, 918
d.rounded_rectangle([LX, ST, RX, SB], radius=22, fill=PANEL_S)
d.rectangle([LX, ST, LX + 8, SB], fill=GREEN)
tracked(d, LX + 40, ST + 30, "SO IT LOST THE SCOREBOARD  ·  LENGTH-MATCHED MACRO-F1", hv(20), GREEN, 2)

d.text((LX + 40, ST + 84), "184M zero-shot NLI model", font=hv(27), fill=SLATE, anchor="lm")
d.text((RX - 38, ST + 84), "0.69", font=ab(44), fill=SLATE, anchor="rm")
d.text((LX + 40, ST + 130), "one-line word-overlap rule", font=hv(27), fill=OFFW, anchor="lm")
d.text((RX - 38, ST + 130), "0.92", font=ab(44), fill=GREEN, anchor="rm")

# --- Footer ---
d.text((LX, 982),
       "HaluEval-QA  ·  20,000 balanced QA samples  ·  length-matched control  ·  zero-shot DeBERTa-v3 NLI",
       font=hv(21), fill=FOOT, anchor="lm")

out = "/Users/anthonyrodrigues/Desktop/YC-Portfolio-Projects/AI-Agent-Conversation-Quality-Scorer/results/posts/post63_social.png"
img.save(out, "PNG")
print("saved", out)
