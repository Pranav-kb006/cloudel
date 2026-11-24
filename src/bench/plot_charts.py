"""Generate 300-DPI publication-quality charts from measured ablation_summary.json."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SUMMARY_PATH = ROOT / "results" / "ablation_summary.json"
CHARTS_DIR = ROOT / "results" / "charts"


def setup_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10.5,
        "axes.edgecolor": "#cbd5e1",
        "axes.linewidth": 0.9,
        "grid.color": "#e2e8f0",
        "grid.linestyle": "--",
        "grid.linewidth": 0.7,
        "figure.facecolor": "white",
        "axes.facecolor": "#f8fafc",
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def plot_bandwidth_sweep(data: dict) -> None:
    rows = data["bandwidth_sweep"]
    bws = [f"{int(r['bandwidth_mbps'])}\nMbps" for r in rows]
    mono = [r["mono_ttft_s"] for r in rows]
    prog = [r["prog_ttft_s"] for r in rows]
    mask = [r["prog_masking_ratio"] * 100.0 for r in rows]
    x = np.arange(len(bws))
    w = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.3))

    b1 = ax1.bar(x - w / 2, mono, w, label="Monolithic Baseline", color="#ef4444", edgecolor="#b91c1c", linewidth=0.8)
    b2 = ax1.bar(x + w / 2, prog, w, label="Progressive Streaming (Ours)", color="#2563eb", edgecolor="#1d4ed8", linewidth=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(bws)
    ax1.set_ylabel("Cold-Start TTFT (seconds)")
    ax1.set_title("(a) Cold-Start TTFT vs. Cloud Network Bandwidth")
    ax1.grid(axis="y", alpha=0.8)
    ax1.set_axisbelow(True)
    ax1.legend(frameon=True, facecolor="white")
    for bar, r in zip(b2, rows):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(mono) * 0.02,
            f"-{r['ttft_reduction_pct']:.0f}%\n({r['speedup_x']:.2f}x)",
            ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#1e3a8a",
        )
    ax1.set_ylim(0, max(mono) * 1.18)

    ax2.plot(x, mask, marker="o", linewidth=2.4, markersize=8, color="#059669", label="I/O Masking Ratio (%)")
    ax2.fill_between(x, mask, alpha=0.14, color="#10b981")
    ax2.set_xticks(x)
    ax2.set_xticklabels(bws)
    ax2.set_ylabel("I/O Masking Ratio (%)")
    ax2.set_ylim(0, 105)
    ax2.set_title("(b) Network I/O Overlap Efficiency Across Bandwidths")
    ax2.grid(True, alpha=0.8)
    for xi, m in zip(x, mask):
        ax2.annotate(f"{m:.1f}%", (xi, m), textcoords="offset points", xytext=(0, 9),
                     ha="center", fontsize=9.5, fontweight="bold", color="#065f46")
    ax2.legend(frameon=True, facecolor="white", loc="lower right")

    fig.tight_layout(pad=2.0)
    fig.savefig(CHARTS_DIR / "fig1_bandwidth_sweep.png")
    plt.close(fig)


def plot_gantt_overlap(data: dict) -> None:
    pair = data["gantt_250mbps"]
    mono = pair["monolithic"]
    prog = pair["progressive"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11.0, 6.4), sharex=True)
    layers = [r["layer"].replace("model.layers.", "L") for r in prog["per_layer"]]
    y = np.arange(len(layers))

    # Top: Monolithic
    ax1.barh(len(layers) / 2 - 0.5, mono["download_s"], left=0.0, height=len(layers) * 0.72,
             color="#fca5a5", edgecolor="#dc2626", alpha=0.65, label="Blocking Full-Checkpoint Download (290 MiB)")
    for i, r in enumerate(mono["per_layer"]):
        ax1.barh(i, r["compute_end_s"] - r["compute_start_s"], left=r["compute_start_s"],
                 height=0.62, color="#7c3aed", edgecolor="#5b21b6",
                 label="Transformer Layer Compute" if i == 0 else "")
    ax1.axvline(mono["ttft_full_s"], color="#dc2626", linestyle="--", linewidth=1.6,
                label=f"Monolithic TTFT = {mono['ttft_full_s']:.2f}s")
    ax1.set_yticks(y[::2])
    ax1.set_yticklabels(layers[::2])
    ax1.set_title(f"(a) Monolithic Baseline (250 Mbps, seq=128) — Zero I/O-Compute Overlap")
    ax1.grid(axis="x", alpha=0.8)
    ax1.legend(loc="lower right", frameon=True, facecolor="white", fontsize=9)

    # Bottom: Progressive Streaming
    for i, r in enumerate(prog["per_layer"]):
        ax2.barh(i + 0.17, r["fetch_end_s"] - r["fetch_start_s"], left=r["fetch_start_s"],
                 height=0.34, color="#3b82f6", edgecolor="#1d4ed8",
                 label="Async HTTP Range Fetch (Layer i)" if i == 0 else "")
        ax2.barh(i - 0.17, r["compute_end_s"] - r["compute_start_s"], left=r["compute_start_s"],
                 height=0.34, color="#10b981", edgecolor="#047857",
                 label="Transformer Layer Compute (Layer i)" if i == 0 else "")
    ax2.axvline(prog["ttft_full_s"], color="#059669", linestyle="--", linewidth=1.6,
                label=f"Progressive TTFT = {prog['ttft_full_s']:.2f}s (Masking {prog['masking_ratio']*100:.1f}%)")
    ax2.set_yticks(y[::2])
    ax2.set_yticklabels(layers[::2])
    ax2.set_xlabel("Wall-Clock Time Since Cold Invocation (seconds)")
    ax2.set_title(f"(b) Progressive Layer-Streaming Engine (K=2, Evicting) — Concurrent Fetch & Compute")
    ax2.grid(axis="x", alpha=0.8)
    ax2.legend(loc="lower right", frameon=True, facecolor="white", fontsize=9)

    fig.tight_layout(pad=1.6)
    fig.savefig(CHARTS_DIR / "fig2_gantt_overlap.png")
    plt.close(fig)


def plot_rss_trajectory(data: dict) -> None:
    pair = data["gantt_250mbps"]
    mono = pair["monolithic"]
    prog = pair["progressive"]

    m_t = [p[0] for p in mono["rss_trace"]]
    m_base = mono["baseline_rss_mb"]
    m_delta = [max(0.0, p[1] - m_base) for p in mono["rss_trace"]]

    p_t = [p[0] for p in prog["rss_trace"]]
    p_base = prog["baseline_rss_mb"]
    p_delta = [max(0.0, p[1] - p_base) for p in prog["rss_trace"]]

    fig, ax = plt.subplots(figsize=(9.2, 4.3))
    ax.plot(m_t, m_delta, color="#dc2626", linewidth=2.3, label=f"Monolithic Baseline (Peak ΔRSS = {mono['delta_rss_mb']:.1f} MB)")
    ax.fill_between(m_t, m_delta, alpha=0.12, color="#ef4444")
    ax.plot(p_t, p_delta, color="#2563eb", linewidth=2.3, label=f"Progressive Streaming + Eviction (Peak ΔRSS = {prog['delta_rss_mb']:.1f} MB)")
    ax.fill_between(p_t, p_delta, alpha=0.16, color="#3b82f6")

    ax.set_xlabel("Elapsed Time Since Cold Invocation (seconds)")
    ax.set_ylabel("Incremental Working-Set RSS Memory (MB)")
    ax.set_title("Working-Set Memory Footprint Over Time (250 Mbps, 12-Layer Llama BF16)")
    ax.grid(True, alpha=0.8)
    ax.legend(frameon=True, facecolor="white", loc="upper right")
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "fig3_rss_memory_trajectory.png")
    plt.close(fig)


def plot_ablation_dual(data: dict) -> None:
    la = data["lookahead_ablation"]
    sq = data["seqlen_scaling"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.3))

    # (a) Lookahead K vs TTFT & Delta RSS
    ks = [r["lookahead_k"] for r in la]
    ttfts = [r["ttft_full_s"] for r in la]
    drss = [r["delta_rss_mb"] for r in la]

    l1 = ax1.plot(ks, ttfts, marker="s", linewidth=2.2, markersize=7, color="#2563eb", label="Cold-Start TTFT (s)")
    ax1.set_xticks(ks)
    ax1.set_xlabel("Prefetch Lookahead Depth (K layers)")
    ax1.set_ylabel("Cold-Start TTFT (seconds)", color="#1d4ed8")
    ax1.tick_params(axis="y", labelcolor="#1d4ed8")
    ax1.grid(True, alpha=0.8)
    ax1.set_title("(a) Lookahead Depth Ablation (250 Mbps, seq=128)")

    ax1b = ax1.twinx()
    l2 = ax1b.plot(ks, drss, marker="^", linestyle="--", linewidth=2.0, markersize=7, color="#d97706", label="Peak Working-Set ΔRSS (MB)")
    ax1b.set_ylabel("Peak ΔRSS Memory (MB)", color="#b45309")
    ax1b.tick_params(axis="y", labelcolor="#b45309")
    lines = l1 + l2
    ax1.legend(lines, [l.get_label() for l in lines], loc="upper center", frameon=True, facecolor="white", fontsize=8.8)

    # (b) Sequence Length Scaling vs Masking Ratio & Speedup
    seqs = [str(r["seq_len"]) for r in sq]
    x = np.arange(len(seqs))
    masks = [r["masking_ratio"] * 100.0 for r in sq]
    reds = [r["ttft_reduction_pct"] for r in sq]

    w = 0.35
    ax2.bar(x - w / 2, masks, w, color="#10b981", edgecolor="#047857", label="I/O Masking Ratio (%)")
    ax2.bar(x + w / 2, reds, w, color="#6366f1", edgecolor="#4338ca", label="TTFT Reduction (%)")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"L={s}" for s in seqs])
    ax2.set_xlabel("Input Prompt Length (tokens)")
    ax2.set_ylabel("Percentage (%)")
    ax2.set_ylim(0, 108)
    ax2.set_title("(b) Prompt Length Scaling (500 Mbps, K=2)")
    ax2.grid(axis="y", alpha=0.8)
    ax2.set_axisbelow(True)
    ax2.legend(frameon=True, facecolor="white", fontsize=9)
    for xi, m, r in zip(x, masks, reds):
        ax2.text(xi - w / 2, m + 1.8, f"{m:.0f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#065f46")
        ax2.text(xi + w / 2, r + 1.8, f"{r:.0f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#3730a3")

    fig.tight_layout(pad=2.0)
    fig.savefig(CHARTS_DIR / "fig4_ablation_lookahead_seqlen.png")
    plt.close(fig)


def main() -> int:
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    setup_style()
    data = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    plot_bandwidth_sweep(data)
    plot_gantt_overlap(data)
    plot_rss_trajectory(data)
    plot_ablation_dual(data)
    print(f"Generated 4 publication charts in {CHARTS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
