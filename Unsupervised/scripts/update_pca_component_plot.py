from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    csv_path = root / "Unsupervised" / "outputs" / "kmeans_benchmark" / "unsupervised_experiments_results.csv"
    out_path = root / "Unsupervised" / "outputs" / "kmeans_benchmark" / "figures" / "pca_component_effects.png"

    df = pd.read_csv(csv_path)
    pca = df[df["family"] == "kmeans_pca"].copy()
    pca["components"] = pca["variant"].str.extract(r"(\d+)").astype(int)
    pca = pca.sort_values("components", ascending=True)
    components_min = int(pca["components"].min())
    components_max = int(pca["components"].max())
    xticks_int = list(range(components_min, components_max + 1))

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    metrics = [
        ("silhouette", "Silhouette (higher is better)"),
        ("davies_bouldin", "Davies-Bouldin (lower is better)"),
        ("calinski_harabasz", "Calinski-Harabasz (higher is better)"),
        ("normalized_mutual_info", "NMI to burnout quartiles (higher is better)"),
    ]

    for ax, (col, title) in zip(axes.ravel(), metrics):
        ax.plot(pca["components"], pca[col], marker="o", linewidth=2)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("PCA components kept")
        ax.set_xticks(xticks_int)
        ax.set_xticklabels([str(v) for v in xticks_int])
        ax.grid(alpha=0.25)

    fig.suptitle("How PCA component count changes K-Means quality", fontsize=13, y=1.02)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
