import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from Classification3D.utils import LOGS_DIR


DEFAULT_CSV_PATH = Path(LOGS_DIR) / "analysis_summary.csv"
DEFAULT_OUTPUT_PATH = Path(LOGS_DIR) / "incor_accuracy_vs_training_time.png"


def plot_incor_accuracy_vs_training_time(
    csv_path: str | os.PathLike = DEFAULT_CSV_PATH,
    output_path: str | os.PathLike = DEFAULT_OUTPUT_PATH,
) -> pd.DataFrame:
    """Plot mean test accuracy against mean training time for Incor groups."""
    data = pd.read_csv(csv_path)

    required_columns = {
        "Dataset",
        "Model",
        "Group",
        "Train Time (s)_mean",
        "Test Accuracy_mean",
    }
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Colunas ausentes no CSV: {missing}")

    incor = data[data["Dataset"].str.lower() == "incor"].copy()
    if incor.empty:
        raise ValueError("Nenhum grupo do dataset 'incor' foi encontrado.")

    incor["Train Time (min)"] = incor["Train Time (s)_mean"] / 60.0
    incor["Stopping"] = incor["Group"].str.extract(
        r"_(full|earlyStop)(?:_|$)", expand=False
    ).fillna("unknown")

    colors = {
        model: color
        for model, color in zip(
            sorted(incor["Model"].unique()),
            ("#1f77b4", "#ff7f0e", "#2ca02c"),
        )
    }
    markers = {"full": "o", "earlyStop": "s", "unknown": "^"}

    figure, axis = plt.subplots(figsize=(14, 9))
    for (model, stopping), subset in incor.groupby(["Model", "Stopping"]):
        axis.scatter(
            subset["Train Time (min)"],
            subset["Test Accuracy_mean"],
            color=colors[model],
            marker=markers[stopping],
            s=75,
            alpha=0.85,
            label=f"{model} - {stopping}",
        )

        # for _, row in subset.iterrows():
        #     axis.annotate(
        #         row["Group"],
        #         (row["Train Time (min)"], row["Test Accuracy_mean"]),
        #         xytext=(5, 5),
        #         textcoords="offset points",
        #         fontsize=7,
        #         alpha=0.8,
        #     )

    axis.set_title("Incor: acurácia em função do tempo médio de treinamento")
    axis.set_xlabel("Tempo médio de treinamento (minutos)")
    axis.set_ylabel("Acurácia média no teste")
    axis.set_ylim(0.6, 0.9)
    axis.grid(alpha=0.25)
    axis.legend(title="Modelo e treinamento")
    figure.tight_layout()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    print(f"Gráfico salvo em: {output}")

    return incor


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plota a acurácia contra o tempo de treinamento dos grupos Incor."
    )
    parser.add_argument(
        "--csv",
        default=str(DEFAULT_CSV_PATH),
        help="Caminho para analysis_summary.csv.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Caminho do PNG de saída.",
    )
    args = parser.parse_args()
    plot_incor_accuracy_vs_training_time(args.csv, args.output)


if __name__ == "__main__":
    main()
