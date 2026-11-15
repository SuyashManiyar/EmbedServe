"""Plot evaluation reports from scripts/evaluate.py and write a Markdown summary.

Two-tier runs become curves against hot-tier size, one per hot-tier backend,
row format and warmup setting; FP32 and INT8 runs become reference lines and
points. Repeats of one configuration are merged: latency statistics are the
median across repeats, and the minimum and maximum are kept for error bars and
the table.

With --memory (a report from scripts/measure_memory.py), process memory is
added to the summary and plotted against lookup latency. With --single-request
(an evaluate.py report run at batch size 1), a per-request table is added. With
--serving (reports from scripts/benchmark_serving.py), request latency is
plotted against request rate with and without dynamic batching.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.ticker  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

MB = 1e6
LATENCIES = ("lookup_batch_latency", "inference_batch_latency")
BACKEND_NAMES = {"python": "Python", "cpp": "C++"}
FORMAT_NAMES = {"fp32": "FP32", "int8": "INT8"}


def tier_format(run: dict) -> str | None:
    return (run.get("row_format") or "fp32") if run["mode"] == "two-tier" else None


def load_runs(paths: list[Path]) -> list[dict]:
    """Load every run and merge repeats of the same configuration."""
    groups: dict[tuple, list[dict]] = {}
    for path in paths:
        report = json.loads(path.read_text())
        for run in report["runs"]:
            run["total_rows"] = sum(run["table_sizes"])
            run["row_format"] = tier_format(run)
            if run["mode"] == "two-tier":
                run.setdefault("hot_backend", "python")
            key = (run["mode"], run.get("hot_backend"), run["row_format"], run["warmup_batches"],
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


def load_memory(path: Path) -> dict[tuple, dict]:
    """Median peak memory per configuration, keyed by (mode, format, capacity, mmap)."""
    groups: dict[tuple, list[dict]] = {}
    for run in json.loads(path.read_text())["runs"]:
        key = (run["mode"], run["row_format"], run["hot_capacity"], run["mmap_reads"])
        groups.setdefault(key, []).append(run)
    memory = {}
    for key, repeats in groups.items():
        memory[key] = {
            "footprint": statistics.median(r["peak_footprint_bytes"] for r in repeats),
            "rss": statistics.median(r["peak_rss_bytes"] for r in repeats),
            "payload": repeats[0]["embedding_payload_bytes"],
            "repeats": len(repeats),
        }
    return memory


def yerr(runs: list[dict], key: str, stat: str):
    """Asymmetric error bars from the range across repeats, or None for single runs."""
    if all(r["repeats"] == 1 for r in runs):
        return None
    lo = [r[key][stat] - r[key][stat + "_range"][0] for r in runs]
    hi = [r[key][stat + "_range"][1] - r[key][stat] for r in runs]
    return [lo, hi]


def split(runs: list[dict]):
    """Baselines by mode; two-tier sweeps keyed by (hot backend, row format, warmup batches)."""
    baselines = {r["mode"]: r for r in runs if r["mode"] != "two-tier"}
    sweeps: dict[tuple[str, str, int], list[dict]] = {}
    for r in runs:
        if r["mode"] == "two-tier":
            sweeps.setdefault((r["hot_backend"], r["row_format"], r["warmup_batches"]), []).append(r)
    for rs in sweeps.values():
        rs.sort(key=lambda r: r["hot_capacity"])
    return baselines, dict(sorted(sweeps.items()))


def pct(r: dict) -> float:
    return 100 * r["hot_capacity"] / r["total_rows"]


def tier_name(backend: str, row_format: str) -> str:
    return f"{BACKEND_NAMES.get(backend, backend)} hot tier, {FORMAT_NAMES[row_format]} rows"


def warmest(sweeps):
    """The most-warmed sweep of each (backend, row format), to keep latency plots readable."""
    best: dict[tuple[str, str], tuple[str, str, int]] = {}
    for key in sweeps:
        if key[:2] not in best or key[2] > best[key[:2]][2]:
            best[key[:2]] = key
    return {key: sweeps[key] for key in best.values()}


def finish(fig, ax, out: Path, legend_size: float = 8) -> None:
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=legend_size)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_hit_rate(sweeps, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    first = next(iter(sweeps))[:2]  # hit rate depends only on the row capacity
    for key, rs in sweeps.items():
        if key[:2] == first:
            ax.plot([pct(r) for r in rs], [100 * r["hit_rate"] for r in rs], "o-",
                    label="no warmup" if key[2] == 0 else f"{key[2]} warmup batches")
    ax.set_xscale("symlog", linthresh=0.1)
    ax.set_xlim(left=0)
    ax.set_xlabel("hot-tier size (% of embedding rows)")
    ax.set_ylabel("hit rate (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Hit rate vs hot-tier size")
    finish(fig, ax, out)


def plot_latency(baselines, sweeps, out: Path, key: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for sweep_key, rs in warmest(sweeps).items():
        xs = [pct(r) for r in rs]
        name = tier_name(*sweep_key[:2])
        line = ax.errorbar(xs, [r[key]["p50_ms"] for r in rs], yerr=yerr(rs, key, "p50_ms"),
                           fmt="o-", capsize=2, label=f"{name}, p50")
        ax.errorbar(xs, [r[key]["p99_ms"] for r in rs], yerr=yerr(rs, key, "p99_ms"),
                    fmt="s--", capsize=2, alpha=0.6, color=line.lines[0].get_color(),
                    label=f"{name}, p99")
    for mode, style in (("fp32", ":"), ("int8", "-.")):
        if mode in baselines:
            ax.axhline(baselines[mode][key]["p50_ms"], color="gray", ls=style,
                       label=f"{mode.upper()} in RAM, p50")
    ax.set_xscale("symlog", linthresh=0.1)
    ax.set_xlim(left=0)
    ax.set_yscale("log")
    ax.set_xlabel("hot-tier size (% of embedding rows)")
    ax.set_ylabel("batch latency (ms, log scale)")
    ax.set_title(title)
    finish(fig, ax, out, legend_size=6.5)


def plot_memory_auc(baselines, sweeps, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for mode, marker in (("fp32", "o"), ("int8", "s")):
        if mode in baselines:
            r = baselines[mode]
            ax.scatter(r["embedding_payload_bytes"] / MB, r["auc"], marker=marker, s=60,
                       label=f"{mode.upper()} in RAM", zorder=3)
    seen = set()
    for (backend, row_format, _), rs in warmest(sweeps).items():
        if row_format in seen:
            continue
        seen.add(row_format)
        ax.scatter([r["embedding_payload_bytes"] / MB for r in rs], [r["auc"] for r in rs],
                   marker="^" if row_format == "fp32" else "v", s=40,
                   label=f"two-tier, {FORMAT_NAMES[row_format]} rows (hot tier only)", zorder=3)
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlabel("resident embedding payload (MB)")
    ax.set_ylabel("validation AUC")
    ax.set_title("Embedding memory vs AUC")
    ax.ticklabel_format(axis="y", useOffset=False)
    finish(fig, ax, out)


def memory_points(baselines, sweeps, memory):
    """(label, [(process MB above baseline, lookup p50 ms, annotation)]) per store family."""
    base = memory[("none", None, None, None)]["footprint"]
    series = []
    for mode in ("fp32", "int8"):
        key = (mode, None, None, None)
        if mode in baselines and key in memory:
            series.append((f"{mode.upper()} in RAM", [(
                (memory[key]["footprint"] - base) / MB,
                baselines[mode]["lookup_batch_latency"]["p50_ms"], "")]))
    for (backend, row_format, _), rs in warmest(sweeps).items():
        if backend != "cpp":
            continue
        points = []
        for r in rs:
            key = ("two-tier", row_format, r["hot_capacity"], True)
            if key in memory:
                points.append(((memory[key]["footprint"] - base) / MB,
                               r["lookup_batch_latency"]["p50_ms"], f"{pct(r):.0f}%"))
        if points:
            series.append((f"two-tier, {FORMAT_NAMES[row_format]} rows", points))
    return series


def plot_memory_latency(series, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for label, points in series:
        xs, ys, notes = zip(*points)
        ax.plot(xs, ys, "o-" if len(points) > 1 else "s", markersize=7, label=label)
        for x, y, note in points:
            if note:
                ax.annotate(note, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=7)
    ax.set_xscale("log")
    ticks = [t for t in (10, 20, 50, 100, 200, 500) if any(
        0.7 * min(x for _, ps in series for x, _, _ in ps) <= t <= 1.3 * max(
            x for _, ps in series for x, _, _ in ps) for _ in [0])]
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("process memory added by the store (MB, log scale)")
    ax.set_ylabel("batch lookup p50 (ms)")
    ax.set_ylim(bottom=0)
    ax.set_title("Memory vs lookup latency (labels: hot-tier size)")
    finish(fig, ax, out)


def latency_table(baselines, sweeps) -> str:
    lines = [
        "| Store | Hot tier | Backend | Rows | Warmup | Runs | AUC | Log loss | Payload (MB) | "
        "Hit rate | Lookup p50 / p99 (ms) | Lookup p99 range (ms) | Inference p50 / p99 (ms) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    def row(r, store, hot, hit, backend="—", row_format="—"):
        lk, inf = r["lookup_batch_latency"], r["inference_batch_latency"]
        lines.append(
            f"| {store} | {hot} | {backend} | {row_format} | {r['warmup_batches']} | "
            f"{r['repeats']} | {r['auc']:.6f} | {r['logloss']:.5f} | "
            f"{r['embedding_payload_bytes'] / MB:.1f} | {hit} | "
            f"{lk['p50_ms']:.2f} / {lk['p99_ms']:.2f} | "
            f"{lk['p99_ms_range'][0]:.2f} – {lk['p99_ms_range'][1]:.2f} | "
            f"{inf['p50_ms']:.2f} / {inf['p99_ms']:.2f} |"
        )

    for mode in ("fp32", "int8"):
        if mode in baselines:
            row(baselines[mode], mode.upper(), "all rows", "—")
    for (backend, row_format, _), rs in sweeps.items():
        for r in rs:
            row(r, "two-tier", f"{r['hot_capacity']:,} ({pct(r):.1f}%)",
                f"{100 * r['hit_rate']:.1f}%", BACKEND_NAMES.get(backend, backend),
                FORMAT_NAMES[row_format])
    return "\n".join(lines)


def memory_table(memory, total_rows: int) -> str:
    base = memory[("none", None, None, None)]["footprint"]
    lines = [
        "| Store | Hot tier | Rows | Reads | Runs | Payload (MB) | Footprint (MB) | "
        "Added by store (MB) | RSS (MB) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for (mode, row_format, capacity, mmap_reads), m in memory.items():
        if mode == "two-tier":
            hot = f"{capacity:,} ({100 * capacity / total_rows:.1f}%)"
            reads = "mmap" if mmap_reads else "block cache"
        else:
            hot, reads = ("—", "—") if mode == "none" else ("all rows", "—")
        store = "no store (baseline)" if mode == "none" else mode
        lines.append(
            f"| {store} | {hot} | {FORMAT_NAMES.get(row_format, '—')} | {reads} | {m['repeats']} | "
            f"{m['payload'] / MB:.1f} | {m['footprint'] / MB:.0f} | "
            f"{(m['footprint'] - base) / MB:.0f} | {m['rss'] / MB:.0f} |")
    return "\n".join(lines)


def store_name(store: dict) -> str:
    if store["mode"] != "two-tier":
        return f"{store['mode'].upper()} in RAM"
    return (f"two-tier, {FORMAT_NAMES[store['row_format']]} rows, "
            f"{store['hot_capacity']:,}-row {BACKEND_NAMES[store['hot_backend']]} hot tier")


def load_serving(paths: list[Path]) -> list[dict]:
    """One entry per (report, max batch size, target rate), with medians across repeats."""
    points = []
    for path in paths:
        report = json.loads(path.read_text())
        groups: dict[tuple, list[dict]] = {}
        for run in report["runs"]:
            groups.setdefault((run["max_batch_size"], run["target_rate"]), []).append(run)
        for (max_batch, rate), repeats in sorted(groups.items()):
            point = {"store": store_name(report["store"]), "max_batch_size": max_batch,
                     "target_rate": rate, "repeats": len(repeats),
                     "max_wait_ms": repeats[0]["max_wait_ms"]}
            for stat in ("achieved_rate", "mean_batch_size", "p50_ms", "p99_ms", "send_lag_p99_ms"):
                point[stat] = statistics.median(r[stat] for r in repeats)
            point["p99_range"] = (min(r["p99_ms"] for r in repeats), max(r["p99_ms"] for r in repeats))
            point["saturated"] = point["achieved_rate"] < 0.95 * rate
            points.append(point)
    return points


def batching_name(point: dict) -> str:
    if point["max_batch_size"] == 1:
        return "no batching"
    return f"dynamic batching (≤{point['max_batch_size']}, {point['max_wait_ms']:g} ms)"


def plot_serving(points: list[dict], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    series: dict[tuple, list[dict]] = {}
    for point in points:
        series.setdefault((point["store"], batching_name(point)), []).append(point)
    stores = list(dict.fromkeys(store for store, _ in series))
    for (store, batching), ps in series.items():
        label = batching if len(stores) == 1 else f"{store.split(',')[0]}: {batching}"
        xs = [pt["target_rate"] for pt in ps]
        line, = ax.plot(xs, [pt["p50_ms"] for pt in ps], "o-", label=f"{label}, p50")
        ax.plot(xs, [pt["p99_ms"] for pt in ps], "s--", alpha=0.6, color=line.get_color(),
                label=f"{label}, p99")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("offered load (requests per second, log scale)")
    ax.set_ylabel("request latency (ms, log scale)")
    ax.set_title("Request latency vs load")
    finish(fig, ax, out, legend_size=6.5)


def serving_table(points: list[dict]) -> str:
    lines = [
        "| Store | Batching | Offered (req/s) | Achieved (req/s) | Runs | Mean batch | "
        "p50 (ms) | p99 (ms) | p99 range (ms) | Send lag p99 (ms) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for pt in points:
        achieved = f"{pt['achieved_rate']:,.0f}" + (" (saturated)" if pt["saturated"] else "")
        lines.append(
            f"| {pt['store']} | {batching_name(pt)} | {pt['target_rate']:,.0f} | {achieved} | "
            f"{pt['repeats']} | {pt['mean_batch_size']:.1f} | {pt['p50_ms']:.2f} | {pt['p99_ms']:.2f} | "
            f"{pt['p99_range'][0]:.2f} – {pt['p99_range'][1]:.2f} | {pt['send_lag_p99_ms']:.2f} |")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("reports", nargs="+", type=Path)
    p.add_argument("--memory", type=Path, help="report from scripts/measure_memory.py")
    p.add_argument("--single-request", type=Path, help="evaluate.py report at batch size 1")
    p.add_argument("--serving", nargs="+", type=Path, default=[],
                   help="reports from scripts/benchmark_serving.py")
    p.add_argument("--out-dir", type=Path, default=Path("experiments/plots"))
    args = p.parse_args()
    runs = load_runs(args.reports)
    baselines, sweeps = split(runs)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if sweeps:
        plot_hit_rate(sweeps, args.out_dir / "hit_rate_vs_hot_size.png")
        plot_latency(baselines, sweeps, args.out_dir / "lookup_latency_vs_hot_size.png",
                     "lookup_batch_latency", "Lookup batch latency vs hot-tier size")
        plot_latency(baselines, sweeps, args.out_dir / "inference_latency_vs_hot_size.png",
                     "inference_batch_latency", "End-to-end batch latency vs hot-tier size")
    plot_memory_auc(baselines, sweeps, args.out_dir / "memory_vs_auc.png")
    sections = ["## Batch serving\n\n" + latency_table(baselines, sweeps)]
    if args.memory:
        memory = load_memory(args.memory)
        plot_memory_latency(memory_points(baselines, sweeps, memory),
                            args.out_dir / "memory_vs_latency.png")
        sections.append("## Process memory\n\n" + memory_table(memory, runs[0]["total_rows"]))
    if args.single_request:
        single = split(load_runs([args.single_request]))
        sections.append("## Single requests (batch size 1)\n\n" + latency_table(*single))
    if args.serving:
        points = load_serving(args.serving)
        plot_serving(points, args.out_dir / "serving_latency_vs_rate.png")
        sections.append("## Request serving\n\n" + serving_table(points))
    summary = "\n\n".join(sections)
    (args.out_dir / "results.md").write_text(summary + "\n")
    print(summary)
    print(f"plots: {args.out_dir}")


if __name__ == "__main__":
    main()
