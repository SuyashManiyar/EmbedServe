"""Plot evaluation reports from scripts/evaluate.py and write a Markdown summary.

Each report is one sweep (one warmup setting). Two-tier runs become curves
against hot-tier size; FP32 and INT8 runs become reference lines and points.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MB = 1e6


def load_runs(paths: list[Path]) -> list[dict]:
    runs = []
    for path in paths:
        report = json.loads(path.read_text())
        for run in report["runs"]:
            run["total_rows"] = sum(run["table_sizes"])
            run["source"] = path.name
            runs.append(run)
    return runs


def split(runs: list[dict]):
    baselines = {r["mode"]: r for r in runs if r["mode"] != "two-tier"}
    sweeps: dict[int, list[dict]] = {}
    for r in runs:
        if r["mode"] == "two-tier":
            sweeps.setdefault(r["warmup_batches"], []).append(r)
    for rs in sweeps.values():
        rs.sort(key=lambda r: r["hot_capacity"])
    return baselines, dict(sorted(sweeps.items()))


def pct(r: dict) -> float:
    return 100 * r["hot_capacity"] / r["total_rows"]


def label(warmup: int) -> str:
    return "two-tier, no warmup" if warmup == 0 else f"two-tier, {warmup} warmup batches"


def plot_hit_rate(sweeps, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for warmup, rs in sweeps.items():
        ax.plot([pct(r) for r in rs], [100 * r["hit_rate"] for r in rs], "o-", label=label(warmup))
    ax.set_xscale("symlog", linthresh=0.1)
    ax.set_xlim(left=0)
    ax.set_xlabel("hot-tier size (% of embedding rows)")
    ax.set_ylabel("hit rate (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Hit rate vs hot-tier size")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_latency(baselines, sweeps, out: Path, key: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for warmup, rs in sweeps.items():
        xs = [pct(r) for r in rs]
        line, = ax.plot(xs, [r[key]["p50_ms"] for r in rs], "o-", label=f"{label(warmup)} p50")
        ax.plot(xs, [r[key]["p99_ms"] for r in rs], "s--", color=line.get_color(),
                label=f"{label(warmup)} p99")
    for mode, style in (("fp32", ":"), ("int8", "-.")):
        if mode in baselines:
            ax.axhline(baselines[mode][key]["p99_ms"], color="gray", ls=style,
                       label=f"{mode.upper()} p99 (in RAM)")
    ax.set_xscale("symlog", linthresh=0.1)
    ax.set_xlim(left=0)
    ax.set_yscale("log")
    ax.set_xlabel("hot-tier size (% of embedding rows)")
    ax.set_ylabel("batch latency (ms, log scale)")
    ax.set_title(title)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_memory_auc(baselines, sweeps, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for mode, marker in (("fp32", "o"), ("int8", "s")):
        if mode in baselines:
            r = baselines[mode]
            ax.scatter(r["embedding_payload_bytes"] / MB, r["auc"], marker=marker, s=60,
                       label=mode.upper(), zorder=3)
    if sweeps:
        rs = next(iter(sweeps.values()))
        ax.scatter([r["embedding_payload_bytes"] / MB for r in rs], [r["auc"] for r in rs],
                   marker="^", s=40, label="two-tier (hot tier only)", zorder=3)
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlabel("resident embedding payload (MB)")
    ax.set_ylabel("validation AUC")
    ax.set_title("Memory vs AUC")
    ax.ticklabel_format(axis="y", useOffset=False)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def table(baselines, sweeps) -> str:
    lines = [
        "| Store | Hot tier | Warmup | AUC | Log loss | Payload (MB) | Hit rate | "
        "Lookup p50 / p99 (ms) | Inference p50 / p99 (ms) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]

    def row(r, store, hot, hit):
        lk, inf = r["lookup_batch_latency"], r["inference_batch_latency"]
        lines.append(
            f"| {store} | {hot} | {r['warmup_batches']} | {r['auc']:.6f} | {r['logloss']:.5f} | "
            f"{r['embedding_payload_bytes'] / MB:.1f} | {hit} | "
            f"{lk['p50_ms']:.2f} / {lk['p99_ms']:.2f} | {inf['p50_ms']:.2f} / {inf['p99_ms']:.2f} |"
        )

    for mode in ("fp32", "int8"):
        if mode in baselines:
            row(baselines[mode], mode.upper(), "all rows", "—")
    for rs in sweeps.values():
        for r in rs:
            row(r, "two-tier", f"{r['hot_capacity']:,} ({pct(r):.1f}%)", f"{100 * r['hit_rate']:.1f}%")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("reports", nargs="+", type=Path)
    p.add_argument("--out-dir", type=Path, default=Path("experiments/plots"))
    args = p.parse_args()
    baselines, sweeps = split(load_runs(args.reports))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if sweeps:
        plot_hit_rate(sweeps, args.out_dir / "hit_rate_vs_hot_size.png")
        plot_latency(baselines, sweeps, args.out_dir / "lookup_latency_vs_hot_size.png",
                     "lookup_batch_latency", "Lookup batch latency vs hot-tier size")
        plot_latency(baselines, sweeps, args.out_dir / "inference_latency_vs_hot_size.png",
                     "inference_batch_latency", "End-to-end batch latency vs hot-tier size")
    plot_memory_auc(baselines, sweeps, args.out_dir / "memory_vs_auc.png")
    summary = table(baselines, sweeps)
    (args.out_dir / "results.md").write_text(summary + "\n")
    print(summary)
    print(f"plots: {args.out_dir}")


if __name__ == "__main__":
    main()
