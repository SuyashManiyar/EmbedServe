"""Plot evaluation reports from scripts/evaluate.py and write a Markdown summary.

Two-tier runs become curves against hot-tier size, one per hot-tier backend and
warmup setting; FP32 and INT8 runs become reference lines and points. Repeats
of one configuration are merged: latency statistics are the median across
repeats, and the minimum and maximum are kept for error bars and the table.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MB = 1e6
LATENCIES = ("lookup_batch_latency", "inference_batch_latency")


def load_runs(paths: list[Path]) -> list[dict]:
    """Load every run and merge repeats of the same configuration."""
    groups: dict[tuple, list[dict]] = {}
    for path in paths:
        report = json.loads(path.read_text())
        for run in report["runs"]:
            run["total_rows"] = sum(run["table_sizes"])
            key = (run["mode"], run.get("hot_backend"), run["warmup_batches"],
                   run.get("hot_capacity"), run["rows"], run["batch_size"])
            groups.setdefault(key, []).append(run)
    merged = []
    for repeats in groups.values():
        run = dict(repeats[0], repeats=len(repeats))
        for name in LATENCIES:
            run[name] = {}
            for stat in ("p50_ms", "p99_ms", "mean_ms"):
                values = [r[name][stat] for r in repeats]
                run[name][stat] = statistics.median(values)
                run[name][stat + "_range"] = (min(values), max(values))
        merged.append(run)
    return merged


def yerr(runs: list[dict], key: str, stat: str):
    """Asymmetric error bars from the range across repeats, or None for single runs."""
    if all(r["repeats"] == 1 for r in runs):
        return None
    lo = [r[key][stat] - r[key][stat + "_range"][0] for r in runs]
    hi = [r[key][stat + "_range"][1] - r[key][stat] for r in runs]
    return [lo, hi]


def split(runs: list[dict]):
    """Baselines by mode; two-tier sweeps keyed by (hot backend, warmup batches)."""
    baselines = {r["mode"]: r for r in runs if r["mode"] != "two-tier"}
    sweeps: dict[tuple[str, int], list[dict]] = {}
    for r in runs:
        if r["mode"] == "two-tier":
            key = (r.get("hot_backend", "python"), r["warmup_batches"])
            sweeps.setdefault(key, []).append(r)
    for rs in sweeps.values():
        rs.sort(key=lambda r: r["hot_capacity"])
    return baselines, dict(sorted(sweeps.items()))


def pct(r: dict) -> float:
    return 100 * r["hot_capacity"] / r["total_rows"]


BACKEND_NAMES = {"python": "Python", "cpp": "C++"}


def label(key: tuple[str, int]) -> str:
    backend, warmup = key
    when = "no warmup" if warmup == 0 else f"{warmup} warmup"
    return f"{BACKEND_NAMES.get(backend, backend)} hot tier, {when}"


def latency_sweeps(sweeps):
    """Latency plots show only the most-warmed sweep of each backend, to stay readable."""
    best: dict[str, tuple[str, int]] = {}
    for backend, warmup in sweeps:
        if backend not in best or warmup > best[backend][1]:
            best[backend] = (backend, warmup)
    return {key: sweeps[key] for key in best.values()}


def plot_hit_rate(sweeps, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    first_backend = next(iter(sweeps))[0]  # backends give identical hit rates
    for key, rs in sweeps.items():
        if key[0] != first_backend:
            continue
        ax.plot([pct(r) for r in rs], [100 * r["hit_rate"] for r in rs], "o-",
                label=label(key).split(" hot tier, ")[1])
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
    for sweep_key, rs in latency_sweeps(sweeps).items():
        xs = [pct(r) for r in rs]
        line = ax.errorbar(xs, [r[key]["p50_ms"] for r in rs], yerr=yerr(rs, key, "p50_ms"),
                           fmt="o-", capsize=2, label=f"{label(sweep_key)} p50")
        ax.errorbar(xs, [r[key]["p99_ms"] for r in rs], yerr=yerr(rs, key, "p99_ms"),
                    fmt="s--", capsize=2, color=line.lines[0].get_color(),
                    label=f"{label(sweep_key)} p99")
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
        "| Store | Hot tier | Backend | Warmup | Runs | AUC | Log loss | Payload (MB) | Hit rate | "
        "Lookup p50 / p99 (ms) | Lookup p99 range (ms) | Inference p50 / p99 (ms) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    def row(r, store, hot, hit, backend="—"):
        lk, inf = r["lookup_batch_latency"], r["inference_batch_latency"]
        lines.append(
            f"| {store} | {hot} | {backend} | {r['warmup_batches']} | {r['repeats']} | "
            f"{r['auc']:.6f} | {r['logloss']:.5f} | "
            f"{r['embedding_payload_bytes'] / MB:.1f} | {hit} | "
            f"{lk['p50_ms']:.2f} / {lk['p99_ms']:.2f} | "
            f"{lk['p99_ms_range'][0]:.2f} – {lk['p99_ms_range'][1]:.2f} | "
            f"{inf['p50_ms']:.2f} / {inf['p99_ms']:.2f} |"
        )

    for mode in ("fp32", "int8"):
        if mode in baselines:
            row(baselines[mode], mode.upper(), "all rows", "—")
    for (backend, _), rs in sweeps.items():
        for r in rs:
            row(r, "two-tier", f"{r['hot_capacity']:,} ({pct(r):.1f}%)",
                f"{100 * r['hit_rate']:.1f}%", BACKEND_NAMES.get(backend, backend))
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
