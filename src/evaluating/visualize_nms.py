import os
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

log = logging.getLogger(__name__)


def plot_nms_visualizations(csv_path: str, output_dir: str):
    """Generate visualization plots for NMS benchmark results."""

    os.makedirs(output_dir, exist_ok=True)

    # load data
    df = pd.read_csv(csv_path)
    assert {"method", "conf", "iou", "f1", "precision", "recall", "nms_time_ms"}.issubset(df.columns)

    # aggregated metrics
    agg_3d = (df.groupby(["method", "conf", "iou"])
                .agg({
                    "f1": "mean",
                    "precision": "mean",
                    "recall": "mean",
                    "nms_time_ms": "mean",
                    "fp": "mean",
                    "fn": "mean"
                })
                .reset_index())

    agg_method = (agg_3d.groupby("method")
                    .agg({
                        "f1": "mean",
                        "precision": "mean",
                        "recall": "mean",
                        "nms_time_ms": "mean"
                    })
                    .sort_values("f1", ascending=False))

    # Compute mAP(50) and mAP(50-95) from CSV precision
    map50 = df[df['iou'] == 0.5].groupby('method')['precision'].mean()
    map5095 = df[df['iou'].isin([0.5, 0.6, 0.7, 0.8])].groupby('method')['precision'].mean()

    # Add to aggregated method table
    agg_method['map_50'] = map50
    agg_method['map_50_95'] = map5095

    log.info("\n=== Grand-Mean Ranking ===\n%s", agg_method.round(3))

    # Plot global stats bar-plots (including mAP metrics)
    for metric in ["f1", "precision", "recall", "nms_time_ms", "map_50", "map_50_95"]:
        order = agg_method.index if metric != "nms_time_ms" else agg_method.sort_values(metric).index
        min_val = agg_method[metric].min()
        max_val = agg_method[metric].max()
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
        plt.savefig(f"{output_dir}/bar_{metric}_zoomed.png")
        plt.close()

    # compute global limits for zoomed plots
    f1_min, f1_max = agg_3d['f1'].min(), agg_3d['f1'].max()
    f1_pad = (f1_max - f1_min) * 0.05
    f1_ymin, f1_ymax = f1_min - f1_pad, f1_max + f1_pad

    # zoomed F1 vs IoU
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=agg_3d, x="iou", y="f1", hue="method", marker="o")
    plt.ylim(f1_ymin, f1_ymax)
    plt.title("Zoomed: F1 Score vs IoU Threshold")
    plt.xlabel("IoU Threshold")
    plt.ylabel("F1 Score")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/f1_vs_iou_zoomed.png")
    plt.close()

    # zoomed F1 vs Confidence
    conf_min, conf_max = agg_3d['conf'].min(), agg_3d['conf'].max()
    conf_pad = (conf_max - conf_min) * 0.05
    conf_xmin, conf_xmax = conf_min - conf_pad, conf_max + conf_pad

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=agg_3d, x="conf", y="f1", hue="method", marker="o")
    plt.xlim(conf_xmin, conf_xmax)
    plt.title("Zoomed: F1 Score vs Confidence Threshold")
    plt.xlabel("Confidence Threshold")
    plt.ylabel("F1 Score")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/f1_vs_conf_zoomed.png")
    plt.close()

    # facet grid: F1 vs Confidence only
    g = sns.FacetGrid(agg_3d, col="method", col_wrap=3, height=3.2, sharey=False, sharex=True)
    g.map(sns.lineplot, "conf", "f1", marker="o")
    g.set_axis_labels("Confidence", "F1")
    g.set_titles("{col_name}")
    plt.subplots_adjust(top=0.9)
    g.fig.suptitle("F1 vs Confidence @ each IoU (lines overlap but share x-axis)")
    plt.savefig(f"{output_dir}/facet_f1_vs_conf.png")
    plt.close()

    # FacetGrid: F1 vs IoU
    g = sns.FacetGrid(agg_3d, col="method", col_wrap=3, height=3.2, sharey=False, sharex=True)
    g.map(sns.lineplot, "iou", "f1", marker="o")
    g.set_axis_labels("IoU Threshold", "F1 Score")
    g.set_titles("{col_name}")
    plt.subplots_adjust(top=0.9)
    g.fig.suptitle("F1 vs IoU @ each Confidence (lines overlap but share x-axis)")
    plt.savefig(f"{output_dir}/facet_f1_vs_iou.png")
    plt.close()

    # zoom utility
    def zoom_line(data, x, y, hue, title, fname):
        ymin, ymax = data[y].min(), data[y].max()
        pad = (ymax - ymin) * 0.04
        plt.figure(figsize=(7, 4))
        sns.lineplot(data=data, x=x, y=y, hue=hue, marker="o")
        plt.ylim(ymin - pad, ymax + pad)
        plt.title(title + " (zoomed)")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{fname}.png")
        plt.close()

    zoom_line(agg_3d, "conf", "f1", "method", "F1 vs Confidence", "zoom_f1_conf")
    zoom_line(agg_3d, "iou", "f1", "method", "F1 vs IoU", "zoom_f1_iou")

    # scatter speed vs F1
    plt.figure(figsize=(6, 4))
    sns.scatterplot(data=agg_3d, x="nms_time_ms", y="f1", hue="method", style="method", s=70)
    plt.title("Speed vs F1 (each conf×iou point)")
    plt.xlabel("NMS time (ms)")
    plt.ylabel("F1")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/scatter_speed_f1.png")
    plt.close()

    # Combined heatmaps for F1, Precision, and mAP metrics
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
        plt.savefig(f"{output_dir}/heatmap_grid_{metric}.png")
        plt.close()

    # combined heatmap for NMS time
    metric = "nms_time_ms"
    methods = agg_3d['method'].unique()
    n_methods = len(methods)
    fig, axes = plt.subplots(1, n_methods, figsize=(4 * n_methods, 4), sharey=True)
    for ax, m in zip(axes, methods):
        pivot = agg_3d[agg_3d.method == m].pivot(index="iou", columns="conf", values=metric)
        sns.heatmap(pivot, annot=True, fmt=".1f", cmap="Reds_r", cbar_kws={"label": "NMS Time (ms)"}, ax=ax)
        ax.set_title(m)
        ax.set_xlabel("Confidence")
        ax.set_ylabel("IoU")
    plt.suptitle("NMS Time (ms) across IoU & Confidence (per Method)", y=1.05)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/heatmap_grid_nms_time.png")
    plt.close()

    log.info("All NMS visualizations saved to %s", output_dir)


def main(cfg):
    """
    Entry point for NMS visualization.
    Generates visual plots for NMS benchmark results based on cfg paths.
    """
    csv_path = cfg.paths.evaluate.nms_results_csv
    output_dir = cfg.paths.evaluate.nms_plots_dir

    log.info("Starting NMS visualization...")
    log.info("Reading metrics from: %s", csv_path)
    log.info("Saving plots to: %s", output_dir)

    plot_nms_visualizations(csv_path, output_dir)

    log.info("NMS visualization completed successfully.")