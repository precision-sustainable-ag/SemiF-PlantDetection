import os
import logging
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

log = logging.getLogger(__name__)


class NMSVisualizer:
    def __init__(self, cfg):
        """
        Initialize NMSVisualizer.
        Uses cfg.paths.evaluate.save_dir to locate the latest multi_scale directory.
        """
        self.cfg = cfg
        self.base_save_dir = Path(cfg.paths.evaluate.save_dir)
        self.latest_dir = self._resolve_latest_multiscale_dir(self.base_save_dir)
        self.csv_path = self.latest_dir / "metrics" / "nms_benchmark_all.csv"
        self.output_dir = self.latest_dir / "plots"

    def _resolve_latest_multiscale_dir(self, base_dir: Path) -> Path:
        """
        Find the most recent multi_scale directory under nms_benchmark.
        - If 'multi_scale' (without number) exists, treat it as version 1.
        - Return the one with the highest numeric suffix.
        """
        parent_dir = base_dir / "nms_benchmark"
        if not parent_dir.exists():
            raise FileNotFoundError(f"No nms_benchmark directory found under {base_dir}")

        candidates = []
        for d in parent_dir.iterdir():
            if d.is_dir() and d.name.startswith("multi_scale"):
                try:
                    suffix = int(d.name.replace("multi_scale", "")) if d.name != "multi_scale" else 1
                except ValueError:
                    continue
                candidates.append((suffix, d))

        if not candidates:
            raise FileNotFoundError(f"No multi_scale directories found under {parent_dir}")

        latest = sorted(candidates, key=lambda x: x[0])[-1][1]
        log.info(f"Resolved latest multi_scale directory: {latest}")
        return latest

    def plot_nms_visualizations(self):
        """Generate visualization plots for NMS benchmark results."""
        os.makedirs(self.output_dir, exist_ok=True)

        df = pd.read_csv(self.csv_path)
        assert {"method", "conf", "iou", "f1", "precision", "recall", "nms_time_ms"}.issubset(df.columns)

        # Aggregate per method × conf × iou
        agg_3d = (df.groupby(["method", "conf", "iou"])
                    .agg({
                        "f1": "mean",
                        "precision": "mean",
                        "recall": "mean",
                        "nms_time_ms": "mean",
                        "fp": "mean",
                        "fn": "mean"
                    }).reset_index())

        # Method-level mean
        agg_method = (agg_3d.groupby("method")
                        .agg({
                            "f1": "mean",
                            "precision": "mean",
                            "recall": "mean",
                            "nms_time_ms": "mean"
                        }).sort_values("f1", ascending=False))

        # Compute mAP(50) and mAP(50–95) from precision
        map50 = df[df['iou'] == 0.5].groupby('method')['precision'].mean()
        map5095 = df[df['iou'].isin([0.5, 0.6, 0.7, 0.8])].groupby('method')['precision'].mean()

        agg_method['map_50'] = map50
        agg_method['map_50_95'] = map5095

        log.info("\n=== Grand-Mean Ranking ===\n%s", agg_method.round(3))

        # Plot bar charts (zoomed) for all metrics
        for metric in ["f1", "precision", "recall", "nms_time_ms", "map_50", "map_50_95"]:
            order = agg_method.index if metric != "nms_time_ms" else agg_method.sort_values(metric).index
            min_val, max_val = agg_method[metric].min(), agg_method[metric].max()
            pad = (max_val - min_val) * 0.1

            plt.figure(figsize=(6, 4))
            sns.barplot(
                x=agg_method.loc[order][metric],
                y=order,
                hue=order,
                dodge=False,
                legend=False,
                palette="viridis"
            )
            plt.xlim(min_val - pad, max_val + pad)
            plt.title(f"Mean {metric.upper()} by Method (Zoomed)")
            plt.xlabel(metric.upper())
            plt.ylabel("Method")
            plt.tight_layout()
            plt.savefig(self.output_dir / f"bar_{metric}_zoomed.png")
            plt.close()

        # Plot line charts
        f1_min, f1_max = agg_3d['f1'].min(), agg_3d['f1'].max()
        f1_pad = (f1_max - f1_min) * 0.05
        f1_ymin, f1_ymax = f1_min - f1_pad, f1_max + f1_pad

        plt.figure(figsize=(10, 6))
        sns.lineplot(data=agg_3d, x="iou", y="f1", hue="method", marker="o")
        plt.ylim(f1_ymin, f1_ymax)
        plt.title("Zoomed: F1 Score vs IoU Threshold")
        plt.xlabel("IoU Threshold")
        plt.ylabel("F1 Score")
        plt.tight_layout()
        plt.savefig(self.output_dir / "f1_vs_iou_zoomed.png")
        plt.close()

        # Zoomed F1 vs Confidence
        conf_min, conf_max = agg_3d['conf'].min(), agg_3d['conf'].max()
        conf_pad = (conf_max - conf_min) * 0.05

        plt.figure(figsize=(10, 6))
        sns.lineplot(data=agg_3d, x="conf", y="f1", hue="method", marker="o")
        plt.xlim(conf_min - conf_pad, conf_max + conf_pad)
        plt.title("Zoomed: F1 Score vs Confidence Threshold")
        plt.xlabel("Confidence Threshold")
        plt.ylabel("F1 Score")
        plt.tight_layout()
        plt.savefig(self.output_dir / "f1_vs_conf_zoomed.png")
        plt.close()

        # FacetGrid: F1 vs Confidence
        g = sns.FacetGrid(agg_3d, col="method", col_wrap=3, height=3.2, sharey=False, sharex=True)
        g.map(sns.lineplot, "conf", "f1", marker="o")
        g.set_axis_labels("Confidence", "F1")
        g.set_titles("{col_name}")
        plt.subplots_adjust(top=0.9)
        g.fig.suptitle("F1 vs Confidence @ each IoU")
        plt.savefig(self.output_dir / "facet_f1_vs_conf.png")
        plt.close()

        # FacetGrid: F1 vs IoU
        g = sns.FacetGrid(agg_3d, col="method", col_wrap=3, height=3.2, sharey=False, sharex=True)
        g.map(sns.lineplot, "iou", "f1", marker="o")
        g.set_axis_labels("IoU Threshold", "F1 Score")
        g.set_titles("{col_name}")
        plt.subplots_adjust(top=0.9)
        g.fig.suptitle("F1 vs IoU @ each Confidence")
        plt.savefig(self.output_dir / "facet_f1_vs_iou.png")
        plt.close()

        # Scatter: Speed vs F1
        plt.figure(figsize=(6, 4))
        sns.scatterplot(data=agg_3d, x="nms_time_ms", y="f1", hue="method", style="method", s=70)
        plt.title("Speed vs F1 (each conf×iou point)")
        plt.xlabel("NMS time (ms)")
        plt.ylabel("F1")
        plt.tight_layout()
        plt.savefig(self.output_dir / "scatter_speed_f1.png")
        plt.close()

        # Heatmaps for f1, precision, map_50, map_50_95
        for metric in ["f1", "precision", "map_50", "map_50_95"]:
            methods = agg_3d['method'].unique()
            n_methods = len(methods)
            cmap = "YlGnBu" if metric in ["f1", "map_50", "map_50_95"] else "Oranges"

            fig, axes = plt.subplots(1, n_methods, figsize=(4 * n_methods, 4), sharey=True)
            for ax, m in zip(axes, methods):
                if metric in agg_3d.columns:
                    pivot = agg_3d[agg_3d.method == m].pivot(index="iou", columns="conf", values=metric)
                else:
                    continue
                sns.heatmap(pivot, annot=True, fmt=".3f", cmap=cmap, cbar_kws={"label": metric.upper()}, ax=ax)
                ax.set_title(m)
                ax.set_xlabel("Confidence")
                ax.set_ylabel("IoU")

            plt.suptitle(f"{metric.upper()} across IoU & Confidence (per Method)", y=1.05)
            plt.tight_layout()
            plt.savefig(self.output_dir / f"heatmap_grid_{metric}.png")
            plt.close()

        # Heatmap for NMS time
        metric = "nms_time_ms"
        fig, axes = plt.subplots(1, n_methods, figsize=(4 * n_methods, 4), sharey=True)
        for ax, m in zip(axes, methods):
            pivot = agg_3d[agg_3d.method == m].pivot(index="iou", columns="conf", values=metric)
            sns.heatmap(pivot, annot=True, fmt=".1f", cmap="Reds_r", cbar_kws={"label": "NMS Time (ms)"}, ax=ax)
            ax.set_title(m)
            ax.set_xlabel("Confidence")
            ax.set_ylabel("IoU")

        plt.suptitle("NMS Time (ms) across IoU & Confidence (per Method)", y=1.05)
        plt.tight_layout()
        plt.savefig(self.output_dir / "heatmap_grid_nms_time.png")
        plt.close()

        log.info("All NMS visualizations saved to %s", self.output_dir)


def main(cfg):
    """
    Entry point for Hydra.
    Automatically finds the latest multi_scale directory and generates plots.
    """
    visualizer = NMSVisualizer(cfg)
    log.info("Starting NMS visualization...")
    log.info("Reading metrics from: %s", visualizer.csv_path)
    log.info("Saving plots to: %s", visualizer.output_dir)

    visualizer.plot_nms_visualizations()

    log.info("NMS visualization completed successfully.")