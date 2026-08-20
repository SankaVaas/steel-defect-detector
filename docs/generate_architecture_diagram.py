"""
generate_architecture_diagram.py — builds docs/architecture.png: the
seven-stage production cascade (data pool -> foundation backbone -> anomaly
triage -> detection -> classification -> confidence-gated decision -> MLOps
loop back to the data pool).

Run once to regenerate the diagram after any architectural change:
    python docs/generate_architecture_diagram.py
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.path import Path as MplPath

COLORS = {
    "gray": ("#F1EFE8", "#5F5E5A", "#2C2C2A"),
    "purple": ("#EEEDFE", "#7F77DD", "#26215C"),
    "teal": ("#E1F5EE", "#1D9E75", "#04342C"),
    "coral": ("#FAECE7", "#D85A30", "#4A1B0C"),
    "amber": ("#FAEEDA", "#BA7517", "#412402"),
    "green": ("#EAF3DE", "#639922", "#173404"),
    "red": ("#FCEBEB", "#E24B4A", "#501313"),
}

fig, ax = plt.subplots(figsize=(9.5, 10.6))
ax.set_xlim(0, 680)
ax.set_ylim(0, 675)
ax.invert_yaxis()
ax.axis("off")
fig.patch.set_facecolor("white")


def box(x, y, w, h, title, subtitle, color_key, fontsize_title=12, fontsize_sub=9.5):
    fill, stroke, text_color = COLORS[color_key]
    fb = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0,rounding_size=8",
        linewidth=1.1, edgecolor=stroke, facecolor=fill, zorder=2,
    )
    ax.add_patch(fb)
    cy = y + h / 2
    if subtitle:
        ax.text(x + w / 2, cy - 8, title, ha="center", va="center",
                 fontsize=fontsize_title, color=text_color, fontweight="bold", zorder=3)
        ax.text(x + w / 2, cy + 10, subtitle, ha="center", va="center",
                 fontsize=fontsize_sub, color=text_color, zorder=3)
    else:
        ax.text(x + w / 2, cy, title, ha="center", va="center",
                 fontsize=fontsize_title, color=text_color, fontweight="bold", zorder=3)
    return (x + w / 2, y, y + h)  # center-x, top-y, bottom-y


def arrow(x1, y1, x2, y2, style="-|>", lw=1.2, color="#5F5E5A", connectionstyle="arc3,rad=0", ls="-"):
    fa = FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle=style, mutation_scale=14,
        linewidth=lw, color=color, connectionstyle=connectionstyle,
        linestyle=ls, zorder=1,
    )
    ax.add_patch(fa)


BOX_W = 300
X0 = 190

# Row 1: Data pool
c1, top1, bot1 = box(X0, 20, BOX_W, 60, "Data pool",
                      "Unlabeled + labeled steel-strip images", "gray")

# Row 2: Foundation backbone
c2, top2, bot2 = box(X0, 115, BOX_W, 60, "Foundation backbone",
                      "Self-supervised pretraining (MAE / DINO)", "purple")
arrow(c1, bot1, c2, top2)

# Row 3: Anomaly triage
c3, top3, bot3 = box(X0, 210, BOX_W, 60, "Real-time anomaly triage",
                      "Every frame, line speed, flags ROIs", "teal")
arrow(c2, bot2, c3, top3)

# Row 4: Detection + localization
c4, top4, bot4 = box(X0, 305, BOX_W, 60, "Detection + localization",
                      "Runs only on flagged regions", "coral")
arrow(c3, bot3, c4, top4)

# Row 5: Classification + calibration
c5, top5, bot5 = box(X0, 400, BOX_W, 60, "Classify + calibrate",
                      "Type, severity, calibrated confidence", "amber")
arrow(c4, bot4, c5, top5)

# Row 6: Branch — auto accept/reject vs human review
LEFT_X, RIGHT_X = 110, 380
BRANCH_W = 200
c6l, top6l, bot6l = box(LEFT_X, 495, BRANCH_W, 60, "Auto accept / reject",
                         "Line control system", "green")
c6r, top6r, bot6r = box(RIGHT_X, 495, BRANCH_W, 60, "Human review queue",
                         "Metallurgist verifies", "red")
arrow(c5, bot5, c6l, top6l)
arrow(c5, bot5, c6r, top6r)
ax.text((c5 + c6l) / 2 - 18, (bot5 + top6l) / 2 - 6, "confident", fontsize=9, color="#5F5E5A", ha="center")
ax.text((c5 + c6r) / 2 + 18, (bot5 + top6r) / 2 - 6, "uncertain", fontsize=9, color="#5F5E5A", ha="center")

# Row 7: MLOps drift monitor / retrain trigger
c7, top7, bot7 = box(X0, 590, BOX_W, 60, "Drift monitor + retrain trigger",
                      "MLOps: versioning, shadow deploy", "gray")
arrow(c6l, bot6l, c7, top7)
arrow(c6r, bot6r, c7, top7)

# Feedback loop back to data pool, routed along the right margin
loop_x = 560
verts = [
    (X0 + BOX_W, (top7 + bot7) / 2),
    (loop_x, (top7 + bot7) / 2),
    (loop_x, (top1 + bot1) / 2),
    (X0 + BOX_W, (top1 + bot1) / 2),
]
codes = [MplPath.MOVETO, MplPath.LINETO, MplPath.LINETO, MplPath.LINETO]
path = MplPath(verts, codes)
patch = FancyArrowPatch(
    path=path, arrowstyle="-|>", mutation_scale=14, linewidth=1.1,
    color="#888780", linestyle=(0, (4, 3)), zorder=1,
)
ax.add_patch(patch)
ax.text(loop_x + 14, (top1 + bot7) / 2, "feeds new labels\nback into training",
        fontsize=9, color="#5F5E5A", ha="left", va="center", rotation=90)

fig.tight_layout(pad=1.5)
fig.savefig("/home/claude/steel-defect-detector/docs/architecture.png", dpi=200, bbox_inches="tight")
print("Saved docs/architecture.png")
