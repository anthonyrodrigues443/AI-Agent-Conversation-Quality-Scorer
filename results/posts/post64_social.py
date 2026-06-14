"""
Designed social asset for Post #64 — AI-Agent-Conversation-Quality-Scorer, Phase 3.
Claim carried: the sneakiest hallucination quotes its source almost word-for-word and
flips a single fact. A word-overlap detector caught 0 of the 175 near-verbatim fakes;
a fine-tuned 22M cross-encoder that reads the claim in context rescued 96% of them.

PIL-only (no matplotlib). 1080x1080, dark navy, typography-driven before/after card.
Every number on the graphic (0 / 175, 96%) appears verbatim in the post body.
"""
from PIL import Image, ImageDraw, ImageFont

W = H = 1080
BG      = (15, 23, 42)      # #0F172A slate-900
OFFW    = (250, 250, 249)   # #FAFAF9 off-white
SLATE   = (148, 163, 184)   # #94A3B8 muted label
ROSE    = (251, 113, 133)   # #FB7185 the miss
GREEN   = (52, 211, 153)    # #34D399 the rescue
SKY     = (56, 189, 248)    # #38BDF8 eyebrow
PANEL   = (24, 35, 56)      # slate-800-ish card fill (the rescue)
PANEL_R = (40, 24, 33)      # dark-rose card fill (the miss)
PANEL_S = (17, 28, 30)      # dark-slate takeaway strip
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
tracked(d, LX, 96, "HALLUCINATION DETECTION   ·   THE NEAR-VERBATIM FAKE", hv(21), SKY, 3)

# --- Headline (the post's opening claim) ---
hf = ab(76)
d.text((LX, 150), "A word-perfect quote.", font=hf, fill=OFFW, anchor="lt")
d.text((LX, 240), "One fact flipped.", font=hf, fill=OFFW, anchor="lt")

# context line under headline
d.text((LX, 324), "These fakes look perfectly grounded. Here is how many a",
       font=hv(24), fill=SLATE, anchor="lt")
d.text((LX, 357), "word-overlap detector caught, against a model that reads context:",
       font=hv(24), fill=SLATE, anchor="lt")

# --- Before/after hero: two panels ---
PT, PB = 416, 706
midgap = 24
lp_x0, lp_x1 = LX, (LX + RX) // 2 - midgap // 2
rp_x0, rp_x1 = (LX + RX) // 2 + midgap // 2, RX

d.rounded_rectangle([lp_x0, PT, lp_x1, PB], radius=22, fill=PANEL_R)
d.rectangle([lp_x0, PT, lp_x0 + 8, PB], fill=ROSE)     # accent bar on the miss panel
d.rounded_rectangle([rp_x0, PT, rp_x1, PB], radius=22, fill=PANEL)
d.rectangle([rp_x0, PT, rp_x0 + 8, PB], fill=GREEN)    # accent bar on the rescue panel

def panel_block(cx, label, num, ncol, sub, subcol):
    d.text((cx, PT + 50),  label, font=hv(22), fill=OFFW, anchor="mm")
    d.text((cx, PT + 150), num,   font=ab(104), fill=ncol, anchor="mm")
    d.text((cx, PT + 244), sub,   font=hv(22), fill=subcol, anchor="mm")

panel_block((lp_x0 + lp_x1) // 2, "WORD-OVERLAP DETECTOR",
            "0 / 175", ROSE, "near-verbatim fakes caught", ROSE)
panel_block((rp_x0 + rp_x1) // 2, "FINE-TUNED CROSS-ENCODER",
            "96%", GREEN, "of those 175 rescued", GREEN)

# --- Takeaway strip ---
ST, SB = 762, 918
d.rounded_rectangle([LX, ST, RX, SB], radius=22, fill=PANEL_S)
d.rectangle([LX, ST, LX + 8, SB], fill=GREEN)
tracked(d, LX + 40, ST + 36, "WHY IT MATTERS", hv(20), GREEN, 2)
d.text((LX + 40, ST + 92), "A confident answer that looks perfectly sourced and is", font=hv(27), fill=OFFW, anchor="lm")
d.text((LX + 40, ST + 126), "wrong on one detail. The failure surface signals can't see.", font=hv(27), fill=OFFW, anchor="lm")

# --- Footer ---
d.text((LX, 982),
       "HaluEval-QA  ·  ~1,054 source-reusing hallucinations  ·  175 near-verbatim misses  ·  fine-tuned 22M MiniLM-L6",
       font=hv(18), fill=FOOT, anchor="lm")

out = "/Users/anthonyrodrigues/Desktop/YC-Portfolio-Projects/AI-Agent-Conversation-Quality-Scorer/results/posts/post64_social.png"
img.save(out, "PNG")

# --- fit checks (printed for verification) ---
checks = {
    "headline1": d.textlength("A word-perfect quote.", font=ab(76)),
    "headline2": d.textlength("One fact flipped.", font=ab(76)),
    "ctx1": d.textlength("word-overlap detector caught, against a model that reads context:", font=hv(24)),
    "labelL": d.textlength("WORD-OVERLAP DETECTOR", font=hv(22)),
    "labelR": d.textlength("FINE-TUNED CROSS-ENCODER", font=hv(22)),
    "numL": d.textlength("0 / 175", font=ab(104)),
    "take1": d.textlength("A confident answer that looks perfectly sourced and is", font=hv(27)),
    "footer": d.textlength("HaluEval-QA  ·  ~1,054 source-reusing hallucinations  ·  175 near-verbatim misses  ·  fine-tuned 22M MiniLM-L6", font=hv(18)),
}
panel_w = lp_x1 - lp_x0
print("saved", out, "| panel_w", panel_w)
for k, v in checks.items():
    lim = (panel_w - 40) if k in ("labelL", "labelR", "numL") else 900
    print(f"{k}: {v:.0f}px" + ("  <-- OVER" if v > lim else ""))
