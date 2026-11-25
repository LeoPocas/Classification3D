import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils import LOGS_DIR

TRAIN_SUBSET_SUFFIX = "_TrainData_30pct"

BASE_WEIGHTS = {
    "w_fn": 2.0,
    "w_cross": 1.5,
    "w_fp": 1.0,
    "lambda_param": 1.0,
}

# Configuração dos 3 CSVs e seus respectivos gráficos
METRIC_CONFIGS = [
    {
        "col": "i_div_simples_min",
        "label": "Divisão Simples (ΔB_rel / T_min)",
        "csv_path": os.path.join(LOGS_DIR, "individual_efficiency_simples_min.csv"),
        "plot_path": os.path.join(LOGS_DIR, "individual_efficiency_simples_min.png"),
    },
    {
        "col": "i_raiz_min",
        "label": "Divisão com Raiz Quadrada [max(0, ΔB_rel) / (1 + λ·√T_min)]",
        "csv_path": os.path.join(LOGS_DIR, "individual_efficiency_raiz_min.csv"),
        "plot_path": os.path.join(LOGS_DIR, "individual_efficiency_raiz_min.png"),
    },
    {
        "col": "i_log_min",
        "label": "Divisão Logarítmica [max(0, ΔB_rel) / (1 + λ·log2(1 + T_min))]",
        "csv_path": os.path.join(LOGS_DIR, "individual_efficiency_log_min.csv"),
        "plot_path": os.path.join(LOGS_DIR, "individual_efficiency_log_min.png"),
    },
]


@dataclass
class ModelMetrics:
    """Métricas agregadas de uma execução ou de um conjunto de execuções."""

    name: str
    fn: float
    fp: float
    cross_err: float
    n_nor: float
    n_hcm: float
    n_dcm: float
    cost_seconds: float

    @property
    def n_total(self) -> float:
        return self.n_nor + self.n_hcm + self.n_dcm


class IndividualEfficiencyEvaluator:
    """
    Calcula métricas individualizadas usando como referência o classificador
    trivial que sempre prediz a classe majoritária (HCM).
    """

    def __init__(
        self,
        w_fn: float = 2.0,
        w_cross: float = 1.5,
        w_fp: float = 1.0,
        lambda_param: float = 1.0,
        hcm_class_index: int = 1,
    ):
        self.w_fn = w_fn
        self.w_cross = w_cross
        self.w_fp = w_fp
        self.lambda_param = lambda_param
        self.max_weight = max(w_fn, w_cross, w_fp)
        self.hcm_class_index = hcm_class_index

    def compute_clinical_benefit(
        self, fn: float, fp: float, cross_err: float, n_total: float
    ) -> float:
        if n_total <= 0:
            raise ValueError("n_total deve ser positivo.")

        weighted_loss = (
            self.w_fn * fn + self.w_cross * cross_err + self.w_fp * fp
        ) / (n_total * self.max_weight)
        return max(0.0, 1.0 - weighted_loss)

    def compute_majority_dummy_benefit(self, metrics: ModelMetrics) -> float:
        fn_dummy = 0.0
        fp_dummy = metrics.n_nor
        cross_dummy = (
            metrics.n_dcm if self.hcm_class_index == 1 else metrics.n_hcm
        )
        return self.compute_clinical_benefit(
            fn_dummy, fp_dummy, cross_dummy, metrics.n_total
        )

    def evaluate_model(self, candidate: ModelMetrics) -> Dict[str, float]:
        if candidate.cost_seconds <= 0:
            raise ValueError(f"Custo computacional deve ser positivo para '{candidate.name}'.")

        b_ref = self.compute_majority_dummy_benefit(candidate)
        b_cand = self.compute_clinical_benefit(
            candidate.fn, candidate.fp, candidate.cross_err, candidate.n_total
        )

        remaining_margin = 1.0 - b_ref
        delta_b_rel = (
            0.0
            if remaining_margin <= 1e-9
            else (b_cand - b_ref) / remaining_margin
        )

        cost_minutes = candidate.cost_seconds / 60.0
        gain_clipped = max(0.0, delta_b_rel)

        # 1. Divisão simples pelo tempo em minutos
        i_div_simples = delta_b_rel / cost_minutes

        # 2. Divisão com raiz quadrada do tempo em minutos (limitada em [0, 1])
        i_raiz = gain_clipped / (1.0 + self.lambda_param * math.sqrt(cost_minutes))

        # 3. Divisão logarítmica do tempo em minutos (limitada em [0, 1])
        i_log = gain_clipped / (1.0 + self.lambda_param * math.log2(1.0 + cost_minutes))

        return {
            "model_name": candidate.name,
            "b_dummy_hcm": round(b_ref, 5),
            "b_candidate": round(b_cand, 5),
            "delta_b_rel": round(delta_b_rel, 5),
            "cost_seconds": round(candidate.cost_seconds, 4),
            "cost_minutes": round(cost_minutes, 5),
            "i_div_simples_min": round(i_div_simples, 5),
            "i_raiz_min": round(i_raiz, 5),
            "i_log_min": round(i_log, 5),
        }


def _technique_from_group(group_name: str) -> str:
    subset_tag = ""
    base_name = group_name
    if group_name.endswith(TRAIN_SUBSET_SUFFIX):
        subset_tag = TRAIN_SUBSET_SUFFIX
        base_name = group_name[: -len(TRAIN_SUBSET_SUFFIX)]

    match = re.match(
        r"^(?P<dataset>incor|mms)_(?P<model>dual|concat|early_channel)_"
        r"(?P<stopping>full|earlyStop)(?:_|$)",
        base_name,
    )
    core_technique = match.group(0).rstrip("_") if match else base_name
    return f"{core_technique}{subset_tag}"


def _preprocessing_label(group_name: str, technique: str) -> str:
    label = group_name
    if technique.endswith(TRAIN_SUBSET_SUFFIX):
        core_tech = technique[: -len(TRAIN_SUBSET_SUFFIX)]
        if label.endswith(TRAIN_SUBSET_SUFFIX):
            label = label[: -len(TRAIN_SUBSET_SUFFIX)]
        label = label.removeprefix(f"{core_tech}_")
    else:
        label = label.removeprefix(f"{technique}_")
    return label or "Baseline_Raw"


def _metrics_from_json(data: dict, name: str) -> Optional[ModelMetrics]:
    results = data.get("results", {})
    matrix = np.asarray(results.get("confusion_matrix", []), dtype=float)
    timing = results.get("timing", {})

    if matrix.ndim != 2 or matrix.shape[0] < 3 or matrix.shape[1] < 3:
        return None

    cost_seconds = float(timing.get("training_seconds", 0.0))
    
    if cost_seconds <= 0 or matrix.sum() <= 0:
        return None

    fn = float(matrix[1:, 0].sum())
    fp = float(matrix[0, 1:].sum())
    cross_err = float(matrix[1, 2] + matrix[2, 1])

    n_nor = float(matrix[0, :].sum())
    n_hcm = float(matrix[1, :].sum())
    n_dcm = float(matrix[2, :].sum())

    return ModelMetrics(name, fn, fp, cross_err, n_nor, n_hcm, n_dcm, cost_seconds)


def load_metrics_dataframe(logs_dir: str = LOGS_DIR) -> pd.DataFrame:
    records: List[dict] = []
    if not os.path.isdir(logs_dir):
        print(f"Diretório de logs não encontrado: {logs_dir}")
        return pd.DataFrame()

    log_root = Path(logs_dir)
    for path in sorted(log_root.rglob("*.json")):
        if path.name.startswith("analysis_"):
            continue

        group_name = path.parent.name
        technique = _technique_from_group(group_name)
        try:
            with path.open("r", encoding="utf-8") as stream:
                data = json.load(stream)

            metrics = _metrics_from_json(data, group_name)
            if metrics is None:
                continue

            records.append(
                {
                    "Group": group_name,
                    "Technique": technique,
                    "Filename": path.name,
                    "metrics": metrics,
                }
            )
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
            print(f"Erro ao ler {path}: {error}")

    return pd.DataFrame(records)


def _aggregate_metrics(group_df: pd.DataFrame, name: str) -> ModelMetrics:
    metrics = list(group_df["metrics"])
    return ModelMetrics(
        name=name,
        fn=float(np.mean([m.fn for m in metrics])),
        fp=float(np.mean([m.fp for m in metrics])),
        cross_err=float(np.mean([m.cross_err for m in metrics])),
        n_nor=float(np.mean([m.n_nor for m in metrics])),
        n_hcm=float(np.mean([m.n_hcm for m in metrics])),
        n_dcm=float(np.mean([m.n_dcm for m in metrics])),
        cost_seconds=float(np.mean([m.cost_seconds for m in metrics])),
    )


def compute_all_individual_metrics(
    logs_dir: str = LOGS_DIR,
    evaluator: Optional[IndividualEfficiencyEvaluator] = None,
) -> pd.DataFrame:
    evaluator = evaluator or IndividualEfficiencyEvaluator(**BASE_WEIGHTS)
    df = load_metrics_dataframe(logs_dir)
    if df.empty:
        print("Nenhum dado válido extraído.")
        return pd.DataFrame()

    rows: List[dict] = []
    for (technique, group_name), group_df in df.groupby(["Technique", "Group"]):
        agg_metrics = _aggregate_metrics(group_df, group_name)
        res = evaluator.evaluate_model(agg_metrics)
        rows.append(
            {
                "Technique": technique,
                "Train_Regime": "30pct" if technique.endswith(TRAIN_SUBSET_SUFFIX) else "100pct",
                "Group": group_name,
                "Preprocessing": _preprocessing_label(group_name, technique),
                "Runs": len(group_df),
                **res,
            }
        )

    return pd.DataFrame(rows)


def plot_index_vs_time(
    df_csv: pd.DataFrame,
    metric_col: str,
    metric_label: str,
    output_path: str,
) -> None:
    """Plota o gráfico de dispersão Índice vs. Tempo (min) para cada CSV gerado."""
    fig, ax = plt.subplots(figsize=(13, 7.5))

    techniques = sorted(df_csv["Technique"].unique())
    cmap = plt.get_cmap("tab20")
    colors = {tech: cmap(i % 20) for i, tech in enumerate(techniques)}

    for tech in techniques:
        sub = df_csv[df_csv["Technique"] == tech]
        marker = "^" if tech.endswith(TRAIN_SUBSET_SUFFIX) else "o"
        ax.scatter(
            sub["cost_minutes"],
            sub[metric_col],
            label=tech,
            color=colors[tech],
            marker=marker,
            s=65,
            alpha=0.85,
            edgecolors="black",
            linewidths=0.5,
        )

    # Destaca os 5 melhores modelos globais desse CSV no gráfico
    top5 = df_csv.sort_values(metric_col, ascending=False).head(5)
    for rank, row in enumerate(top5.itertuples(), start=1):
        val = getattr(row, metric_col)
        ax.annotate(
            f"#{rank} {row.Preprocessing}\n({row.Technique})",
            xy=(row.cost_minutes, val),
            xytext=(8, 4),
            textcoords="offset points",
            fontsize=7.5,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.8),
        )

    if df_csv[metric_col].min() < 0:
        ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.9)

    ax.set_title(f"Eficiência Individual vs. Tempo — {metric_label}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Tempo de Execução (minutos)")
    ax.set_ylabel(f"Índice ({metric_col})")
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(title="Técnica (▲ = 30pct | ● = 100pct)", fontsize=8, loc="best", framealpha=0.9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Gráfico salvo em: {output_path}")


if __name__ == "__main__":
    base_df = compute_all_individual_metrics()
    if base_df.empty:
        raise SystemExit("Nenhum dado válido encontrado.")

    os.makedirs(LOGS_DIR, exist_ok=True)

    for cfg in METRIC_CONFIGS:
        col = cfg["col"]
        csv_path = cfg["csv_path"]
        plot_path = cfg["plot_path"]

        # Ordena por Técnica e pelo índice específico desta versão
        df_sorted = base_df.sort_values(
            ["Technique", col], ascending=[True, False]
        ).reset_index(drop=True)

        df_sorted.to_csv(csv_path, index=False)
        print(f"\nCSV salvo em: {csv_path}")
        print(
            df_sorted[
                ["Technique", "Preprocessing", "b_dummy_hcm", "b_candidate", "cost_minutes", col]
            ].to_string(index=False)
        )

        plot_index_vs_time(
            df_csv=df_sorted,
            metric_col=col,
            metric_label=cfg["label"],
            output_path=plot_path,
        )