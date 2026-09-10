#!/usr/bin/env python3
"""Build docs/assets/tether.gif (~20s) and the research-note PDF."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)

BG = "#0c0f0d"
PANEL = "#151a16"
INK = "#e8e4d6"
MUTED = "#9aa394"
COPPER = "#c4a35a"
MINT = "#7dce9a"
ROSE = "#e07a5f"


def _axes(title: str):
    fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=100)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.06, 0.92, "TETHER", fontsize=22, color=COPPER, fontweight="bold", fontfamily="sans-serif")
    ax.text(0.06, 0.86, title, fontsize=16, color=INK, fontfamily="sans-serif")
    return fig, ax


def _card(ax, x, y, w, h):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01", facecolor=PANEL, edgecolor="#2a332c"))


def frames():
    scenes = []

    fig, ax = _axes("Universe · FDR, not raw correlation")
    _card(ax, 0.06, 0.18, 0.88, 0.58)
    ax.text(0.1, 0.62, "Consumer Staples  ·  10 pairs tested", color=MUTED, fontsize=13)
    ax.text(0.1, 0.48, "3 of 10 survive FDR", color=MINT, fontsize=28, fontweight="bold")
    ax.text(0.1, 0.34, "1 of 4 TETHERED names die after 2024-01-01", color=COPPER, fontsize=16)
    ax.text(0.1, 0.24, "Load KO/PEP. AAPL/MSFT is the negative control.", color=INK, fontsize=13)
    scenes.append(fig)

    fig, ax = _axes("Walk-forward · discover, then trade")
    _card(ax, 0.06, 0.18, 0.42, 0.58)
    ax.text(0.1, 0.62, "IN-SAMPLE", color=MUTED, fontsize=11)
    ax.text(0.1, 0.48, "Sharpe  0.84", color=INK, fontsize=22)
    ax.text(0.1, 0.36, "TETHERED  ·  β frozen", color=MINT, fontsize=14)
    _card(ax, 0.52, 0.18, 0.42, 0.58)
    ax.text(0.56, 0.62, "OUT-OF-SAMPLE", color=MUTED, fontsize=11)
    ax.text(0.56, 0.48, "Sharpe  0.31", color=ROSE, fontsize=22)
    ax.text(0.56, 0.34, "If live is worse, leave it on the Desk.", color=COPPER, fontsize=13)
    scenes.append(fig)

    fig, ax = _axes("AAPL/MSFT · correlation is not a book")
    _card(ax, 0.06, 0.18, 0.88, 0.58)
    ax.text(0.1, 0.62, "BROKEN", color=ROSE, fontsize=26, fontweight="bold")
    ax.text(0.1, 0.48, "Score ~25   ·   half-life ~114 days   ·   desk wants 8–40", color=INK, fontsize=16)
    ax.text(0.1, 0.34, "Shared tech factor. The spread is two directional bets.", color=MUTED, fontsize=14)
    ax.text(0.1, 0.24, "On-time vs a 114-day clock is not evidence.", color=COPPER, fontsize=14)
    scenes.append(fig)

    fig, ax = _axes("Residual risk · borrow and net book β")
    _card(ax, 0.06, 0.18, 0.42, 0.58)
    ax.text(0.1, 0.62, "BORROW", color=MUTED, fontsize=11)
    ax.text(0.1, 0.46, "50 bps / year", color=INK, fontsize=22)
    ax.text(0.1, 0.32, "on short dollar notional", color=MUTED, fontsize=13)
    _card(ax, 0.52, 0.18, 0.42, 0.58)
    ax.text(0.56, 0.62, "NET BOOK β", color=MUTED, fontsize=11)
    ax.text(0.56, 0.48, "held × β / capital", color=INK, fontsize=18)
    ax.text(0.56, 0.32, "Kalman β drift = sneaky index risk", color=COPPER, fontsize=13)
    scenes.append(fig)

    fig, ax = _axes("Research · grounded in the blotter")
    _card(ax, 0.06, 0.18, 0.88, 0.58)
    ax.text(0.1, 0.58, "Why is this BROKEN?", color=INK, fontsize=22)
    ax.text(0.1, 0.42, "Groq only sees this book: p-value, half-life,", color=MUTED, fontsize=14)
    ax.text(0.1, 0.34, "walk-forward Sharpe, net β, and the last stop.", color=MUTED, fontsize=14)
    ax.text(0.1, 0.24, "Universe → Desk → Research. Forty seconds.", color=COPPER, fontsize=14)
    scenes.append(fig)

    return scenes


def write_gif():
    pngs = []
    tmp = ASSETS / "_frames"
    tmp.mkdir(exist_ok=True)
    for i, fig in enumerate(frames()):
        path = tmp / f"f{i}.png"
        fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.15)
        plt.close(fig)
        pngs.append(Image.open(path).convert("P", palette=Image.ADAPTIVE))
    # 5 frames × 4s at duration=4000 ms ≈ 20s
    pngs[0].save(
        ASSETS / "tether.gif",
        save_all=True,
        append_images=pngs[1:],
        duration=4000,
        loop=0,
    )


def write_pdf():
    pdf_path = ROOT / "docs" / "ko-pep-vs-aapl-msft.pdf"
    md = (ROOT / "docs" / "ko-pep-vs-aapl-msft.md").read_text()
    pages = md.split("\n## ")
    # first chunk includes title
    blocks = [pages[0]] + ["## " + p for p in pages[1:]]
    with PdfPages(pdf_path) as pdf:
        for block in blocks:
            fig, ax = plt.subplots(figsize=(8.5, 11), dpi=120)
            fig.patch.set_facecolor("#f7f4ea")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis("off")
            ax.text(0.08, 0.94, "TETHER", color="#8a7a3d", fontsize=11, fontweight="bold")
            y = 0.88
            for line in block.strip().splitlines():
                if not line.strip():
                    y -= 0.018
                    continue
                size = 11
                weight = "normal"
                color = "#1c1c18"
                text = line
                if line.startswith("# "):
                    size, weight, text = 16, "bold", line[2:]
                elif line.startswith("## "):
                    size, weight, text = 13, "bold", line[3:]
                elif line.startswith("- "):
                    text = "• " + line[2:]
                elif line.startswith("|"):
                    size = 8
                ax.text(0.08, y, text[:110], fontsize=size, color=color, fontweight=weight, wrap=True, va="top")
                y -= 0.028 if size < 13 else 0.034
                if y < 0.06:
                    break
            pdf.savefig(fig, facecolor=fig.get_facecolor())
            plt.close(fig)


if __name__ == "__main__":
    write_gif()
    write_pdf()
    print("wrote", ASSETS / "tether.gif")
    print("wrote", ROOT / "docs" / "ko-pep-vs-aapl-msft.pdf")
