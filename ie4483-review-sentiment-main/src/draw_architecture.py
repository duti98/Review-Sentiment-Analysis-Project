"""Black-and-white architecture figure for the report. Usage: python draw_architecture.py "DeBERTa-v3-base" 12 768"""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from data import RESULTS  # noqa: E402

name = sys.argv[1] if len(sys.argv) > 1 else "Encoder"
layers = sys.argv[2] if len(sys.argv) > 2 else "12"
hidden = sys.argv[3] if len(sys.argv) > 3 else "768"

boxes = [
    ("Raw review text", "string, any length"),
    ("Subword tokenizer", "up to 256 token ids + attention mask"),
    ("Token + position embeddings", f"L x {hidden}"),
    (f"{name} encoder", f"{layers} transformer layers, hidden size {hidden}"),
    ("[CLS] (first token) vector", hidden),
    ("Classification head", f"dropout, dense {hidden} to {hidden}, activation, dropout"),
    ("Output layer", f"dense {hidden} to 2, softmax"),
    ("P(negative), P(positive)", "label = 1 if P(positive) > 0.5"),
]

fig, ax = plt.subplots(figsize=(7.2, 7.6))
ax.set_xlim(0, 10)
ax.set_ylim(0, len(boxes) * 1.1 + 0.4)
ax.axis("off")
w, h, x0 = 6.0, 0.78, 0.4
for i, (title, sub) in enumerate(boxes):
    y = (len(boxes) - 1 - i) * 1.1 + 0.4
    ax.add_patch(FancyBboxPatch((x0, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc="white", ec="black", lw=1.1))
    ax.text(x0 + w / 2, y + h * 0.64, title, ha="center", va="center", fontsize=10.5, fontweight="bold")
    ax.text(x0 + w / 2, y + h * 0.27, sub, ha="center", va="center", fontsize=9)
    if i < len(boxes) - 1:
        ax.add_patch(FancyArrowPatch((x0 + w / 2, y), (x0 + w / 2, y - 0.32), arrowstyle="-|>",
                                     mutation_scale=12, color="black", lw=1))

# bracket showing what is fine-tuned, and the 5-fold ensemble note
top = (len(boxes) - 3) * 1.1 + 0.4 + h
bottom = (len(boxes) - 7) * 1.1 + 0.4
ax.plot([x0 + w + 0.25] * 2, [bottom, top], color="black", lw=1)
ax.plot([x0 + w + 0.15, x0 + w + 0.25], [top, top], color="black", lw=1)
ax.plot([x0 + w + 0.15, x0 + w + 0.25], [bottom, bottom], color="black", lw=1)
ax.text(x0 + w + 0.4, (top + bottom) / 2,
        "all weights\nfine-tuned with\ncross-entropy loss\n(AdamW, 3 epochs)\n\nfinal submission:\naverage of 5\nfold models",
        ha="left", va="center", fontsize=9)
fig.tight_layout()
out = RESULTS / "figures" / "architecture.png"
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=200)
print("wrote", out)
