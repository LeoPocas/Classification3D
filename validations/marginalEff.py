import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
from utils import LOGS_DIR

SUMMARY_OUTPUT_PATH = os.path.join(LOGS_DIR, "marginal_efficiency_summary_1.0_v2.csv")
SENSITIVITY_OUTPUT_PATH = os.path.join(LOGS_DIR, "marginal_efficiency_weight_sensitivity_v2.csv")
WEIGHT_SENSITIVITY_PLOT_PATH = os.path.join(LOGS_DIR, "marginal_efficiency_weight_sensitivity_v2.png")
LAMBDA_SENSITIVITY_PLOT_PATH = os.path.join(LOGS_DIR, "marginal_efficiency_lambda_sensitivity_v2.png")
SENSITIVITY_TECHNIQUE = "incor_concat_full"
TRAIN_SUBSET_SUFFIX = "_TrainData_30pct"
WEIGHT_NAMES = ("w_fn", "w_cross", "w_fp", "lambda_param")

BASE_WEIGHTS = {
    "w_fn": 2.0,
    "w_cross": 1.5,
    "w_fp": 1.0,
    "lambda_param": 1.0,
}


@dataclass
class ModelMetrics:
    """Métricas agregadas de uma execução ou de um conjunto de execuções."""

    name: str
    fn: float
    fp: float
    cross_err: float
    n_total: float
    cost_seconds: float


class MarginalEfficiencyEvaluator:
    """Calcula o Índice de Eficiência Marginal Normalizada (I_EMN) relativo ao modelo mais rápido."""

    def __init__(
        self,
        w_fn: float = 2.0,
        w_cross: float = 1.5,
        w_fp: float = 1.0,
        lambda_param: float = 1.0,
    ):
        self.w_fn = w_fn
        self.w_cross = w_cross
        self.w_fp = w_fp
        self.lambda_param = lambda_param
        self.max_weight = max(w_fn, w_cross, w_fp)

    def compute_clinical_benefit(self, metrics: ModelMetrics) -> float:
        if metrics.n_total <= 0:
            raise ValueError(f"n_total deve ser positivo para '{metrics.name}'.")

        weighted_loss = (
            self.w_fn * metrics.fn
            + self.w_cross * metrics.cross_err
            + self.w_fp * metrics.fp
        ) / (metrics.n_total * self.max_weight)
        return max(0.0, 1.0 - weighted_loss)

    def evaluate_candidate(
        self, reference: ModelMetrics, candidate: ModelMetrics
    ) -> Dict[str, float]:
        b_ref = self.compute_clinical_benefit(reference)
        b_cand = self.compute_clinical_benefit(candidate)
        remaining_margin = 1.0 - b_ref
        delta_b_rel = (
            0.0
            if remaining_margin <= 1e-9
            else (b_cand - b_ref) / remaining_margin
        )

        if reference.cost_seconds <= 0 or candidate.cost_seconds <= 0:
            raise ValueError("Os custos computacionais devem ser positivos.")

        # Como 'reference' é o grupo mais rápido, cost_ratio >= 1.0 (cost_penalty >= 1.0)
        cost_ratio = max(1.0, candidate.cost_seconds / reference.cost_seconds)
        cost_penalty = 1.0 + self.lambda_param * math.log2(cost_ratio)

        return {
            "model_name": candidate.name,
            "b_reference": round(b_ref, 5),
            "b_candidate": round(b_cand, 5),
            "delta_b_rel": round(delta_b_rel, 5),
            "cost_ratio": round(cost_ratio, 5),
            "cost_penalty": round(cost_penalty, 5),
            "i_emn": round(delta_b_rel / cost_penalty, 5),
        }


def _technique_from_group(group_name: str) -> str:
    """
    Extrai dataset, técnica, estratégia de treino e regime de dados (ex: _TrainData_30pct).
    """
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
    """Extrai o rótulo limpo do pré-processamento, tratando também o sufixo _TrainData_30pct."""
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

    cost_seconds = float(timing.get("load_data_seconds", 0.0)) + float(
        timing.get("testing_seconds", 0.0)
    )
    n_total = float(matrix.sum())
    if cost_seconds <= 0 or n_total <= 0:
        return None

    fn = float(matrix[1:, 0].sum())
    fp = float(matrix[0, 1:].sum())
    cross_err = float(matrix[1, 2] + matrix[2, 1])
    return ModelMetrics(name, fn, fp, cross_err, n_total, cost_seconds)


def load_metrics_dataframe(logs_dir: str = LOGS_DIR) -> pd.DataFrame:
    """Carrega todas as execuções válidas dos arquivos JSON."""
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
        filename = path.name
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
                    "Filename": filename,
                    "metrics": metrics,
                }
            )
        except (
            OSError,
            json.JSONDecodeError,
            TypeError,
            ValueError,
        ) as error:
            print(f"Erro ao ler {path}: {error}")

    return pd.DataFrame(records)


def _aggregate_metrics(group_df: pd.DataFrame, name: str) -> ModelMetrics:
    metrics = list(group_df["metrics"])
    return ModelMetrics(
        name,
        float(np.mean([metric.fn for metric in metrics])),
        float(np.mean([metric.fp for metric in metrics])),
        float(np.mean([metric.cross_err for metric in metrics])),
        float(np.mean([metric.n_total for metric in metrics])),
        float(np.mean([metric.cost_seconds for metric in metrics])),
    )


def _select_fastest_reference(
    tech_df: pd.DataFrame,
) -> tuple[ModelMetrics, int, list[tuple[str, ModelMetrics, int]]]:
    """
    Agrega todos os grupos da técnica e elege o grupo de menor cost_seconds médio
    como referência computacional. Retorna (referencia, n_runs_ref, todos_os_grupos)
    para que nenhum grupo (nem Baseline_Raw nem a própria referência) fique de fora.
    """
    aggregated: list[tuple[str, ModelMetrics, int]] = []
    for group_name, group_df in tech_df.groupby("Group"):
        aggregated.append(
            (group_name, _aggregate_metrics(group_df, group_name), len(group_df))
        )

    if len(aggregated) < 2:
        raise ValueError("São necessários pelo menos 2 grupos para realizar a comparação.")

    aggregated.sort(key=lambda item: (item[1].cost_seconds, item[0]))
    _, reference_metrics, ref_runs = aggregated[0]
    # Retorna TODOS os grupos (incluindo Baseline_Raw e a própria referência)
    return reference_metrics, ref_runs, aggregated


def calculate_marginal_efficiency(
    logs_dir: str = LOGS_DIR,
    output_path: str = SUMMARY_OUTPUT_PATH,
    evaluator: Optional[MarginalEfficiencyEvaluator] = None,
) -> pd.DataFrame:
    """Avalia todos os grupos (incluindo Baseline_Raw) contra o método mais rápido da técnica."""
    evaluator = evaluator or MarginalEfficiencyEvaluator()
    df = load_metrics_dataframe(logs_dir)
    if df.empty:
        print("Nenhum dado válido extraído.")
        return pd.DataFrame()

    rows: List[dict] = []

    for technique, tech_df in df.groupby("Technique"):
        try:
            reference, ref_runs, all_groups = _select_fastest_reference(tech_df)
        except ValueError as err:
            print(f"Aviso para técnica '{technique}': {err}")
            continue

        for group_name, candidate, cand_runs in all_groups:
            result = evaluator.evaluate_candidate(reference, candidate)
            rows.append(
                {
                    "Technique": technique,
                    "Train_Regime": "30pct" if technique.endswith(TRAIN_SUBSET_SUFFIX) else "100pct",
                    "Group": group_name,
                    "Preprocessing": _preprocessing_label(group_name, technique),
                    "Is_Fastest_Ref": group_name == reference.name,
                    "Reference_Group": reference.name,
                    "Reference_Runs": ref_runs,
                    "Candidate_Runs": cand_runs,
                    "cost_seconds_mean": round(candidate.cost_seconds, 5),
                    **result,
                }
            )

    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary = summary.sort_values(
            ["Technique", "i_emn"], ascending=[True, False]
        ).reset_index(drop=True)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        summary.to_csv(output_path, index=False)
        print(f"Resumo salvo em: {output_path}")

    return summary


def _selected_group_metrics(
    df: pd.DataFrame, technique: str
) -> tuple[ModelMetrics, list[tuple[str, ModelMetrics]]]:
    technique_df = df[df["Technique"] == technique]
    if technique_df.empty:
        raise ValueError(f"Nenhum dado encontrado para a técnica '{technique}'.")

    reference, _, all_groups_with_runs = _select_fastest_reference(technique_df)
    all_candidates = [(name, metrics) for name, metrics, _ in all_groups_with_runs]
    return reference, sorted(all_candidates, key=lambda item: item[0])


def _evaluate_sensitivity(
    reference: ModelMetrics,
    candidates: list[tuple[str, ModelMetrics]],
    technique: str,
) -> pd.DataFrame:
    rows = []
    multipliers = np.arange(0.1, 2.01, 0.1)
    for group_name, candidate in candidates:
        preprocessing = _preprocessing_label(group_name, technique)
        for weight_name in WEIGHT_NAMES:
            for multiplier in multipliers:
                weights = BASE_WEIGHTS.copy()
                weights[weight_name] *= float(multiplier)
                varied_evaluator = MarginalEfficiencyEvaluator(**weights)
                result = varied_evaluator.evaluate_candidate(reference, candidate)
                rows.append(
                    {
                        "Technique": technique,
                        "Reference_Group": reference.name,
                        "Group": group_name,
                        "Preprocessing": preprocessing,
                        "varied_weight": weight_name,
                        "weight_value": weights[weight_name],
                        "weight_multiplier": round(float(multiplier), 2),
                        "i_emn": result["i_emn"],
                    }
                )
    return pd.DataFrame(rows)


def calculate_weight_sensitivity(
    technique: str = SENSITIVITY_TECHNIQUE,
    logs_dir: str = LOGS_DIR,
    output_path: str = SENSITIVITY_OUTPUT_PATH,
) -> pd.DataFrame:
    """Calcula o impacto individual dos pesos sobre o I_EMN usando o mais rápido como referência."""
    df = load_metrics_dataframe(logs_dir)
    if df.empty:
        return pd.DataFrame()

    reference, candidates = _selected_group_metrics(df, technique)
    print(f"Referência (mais rápido) para '{technique}': {reference.name} ({reference.cost_seconds:.4f}s)")
    sensitivity = _evaluate_sensitivity(reference, candidates, technique)
    sensitivity.to_csv(output_path, index=False)
    print(f"Sensibilidade dos pesos salva em: {output_path}")
    return sensitivity


def plot_weight_sensitivity(
    sensitivity: pd.DataFrame,
    output_path: str = WEIGHT_SENSITIVITY_PLOT_PATH,
) -> None:
    """Gera quatro curvas, uma para cada peso, mantendo os demais fixos."""
    technique = sensitivity["Technique"].iloc[0]
    ref_label = _preprocessing_label(sensitivity["Reference_Group"].iloc[0], technique)
    groups = sorted(sensitivity["Preprocessing"].unique())
    colors = dict(zip(groups, plt.get_cmap("tab20").colors))
    figure, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True, sharey=True)

    for axis, weight_name in zip(axes.flat, WEIGHT_NAMES):
        subset = sensitivity[sensitivity["varied_weight"] == weight_name]
        for group in groups:
            group_data = subset[subset["Preprocessing"] == group]
            is_ref = group == ref_label
            axis.plot(
                group_data["weight_multiplier"],
                group_data["i_emn"],
                marker="o" if not is_ref else None,
                linestyle="--" if is_ref else "-",
                linewidth=1.8 if is_ref else 1.5,
                markersize=3,
                label=f"{group} (Ref)" if is_ref else group,
                color="black" if is_ref else colors[group],
            )
        axis.set_title(weight_name)
        axis.set_xlabel("Multiplicador do peso")
        axis.set_ylabel("I_EMN")
        axis.grid(alpha=0.25)

    axes[0, 0].legend(fontsize=8, loc="best")
    figure.suptitle(f"Sensibilidade do I_EMN aos pesos - {technique} (Ref: {ref_label})")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    print(f"Gráfico dos pesos salvo em: {output_path}")


def plot_lambda_sensitivity(
    sensitivity: pd.DataFrame,
    output_path: str = LAMBDA_SENSITIVITY_PLOT_PATH,
) -> None:
    """Gera o gráfico simples variando lambda com os pesos w fixos."""
    technique = sensitivity["Technique"].iloc[0]
    ref_label = _preprocessing_label(sensitivity["Reference_Group"].iloc[0], technique)
    subset = sensitivity[sensitivity["varied_weight"] == "lambda_param"]
    figure, axis = plt.subplots(figsize=(10, 6))
    groups = sorted(subset["Preprocessing"].unique())
    colors = dict(zip(groups, plt.get_cmap("tab20").colors))
    for group in groups:
        group_data = subset[subset["Preprocessing"] == group]
        is_ref = group == ref_label
        axis.plot(
            group_data["weight_value"],
            group_data["i_emn"],
            marker="o" if not is_ref else None,
            linestyle="--" if is_ref else "-",
            linewidth=1.8 if is_ref else 1.5,
            markersize=4,
            label=f"{group} (Ref)" if is_ref else group,
            color="black" if is_ref else colors[group],
        )
    axis.set_title(f"Impacto de lambda no I_EMN - {technique} (Ref: {ref_label})")
    axis.set_xlabel("Lambda")
    axis.set_ylabel("I_EMN")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8, loc="best")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    print(f"Gráfico de lambda salvo em: {output_path}")


def calculate_2d_interaction(
    technique: str = SENSITIVITY_TECHNIQUE,
    logs_dir: str = LOGS_DIR,
    output_path: str = os.path.join(LOGS_DIR, "phase_diagram_interaction.png"),
) -> pd.DataFrame:
    """
    Avalia a interação simultânea entre Peso Clínico (w_fn / w_fp) e Custo (lambda).
    Gera um Diagrama de Fases mostrando qual modelo vence em cada região.
    """
    df = load_metrics_dataframe(logs_dir)
    if df.empty:
        return pd.DataFrame()

    reference, candidates = _selected_group_metrics(df, technique)
    ref_label = _preprocessing_label(reference.name, technique)

    w_fn_values = np.linspace(1.0, 10.0, 40)
    lambda_values = np.linspace(0.2, 3.0, 40)

    winner_grid = np.zeros((len(lambda_values), len(w_fn_values)), dtype=int)
    margin_grid = np.zeros((len(lambda_values), len(w_fn_values)), dtype=float)

    candidate_names = [_preprocessing_label(name, technique) for name, _ in candidates]
    records = []

    for i, lam in enumerate(lambda_values):
        for j, w_fn in enumerate(w_fn_values):
            w_cross = 1.0 + 0.25 * (w_fn - 1.0)
            evaluator = MarginalEfficiencyEvaluator(
                w_fn=float(w_fn),
                w_cross=float(w_cross),
                w_fp=1.0,
                lambda_param=float(lam),
            )

            scores = []
            for idx, (group_name, cand_metrics) in enumerate(candidates):
                res = evaluator.evaluate_candidate(reference, cand_metrics)
                scores.append(res["i_emn"])
                records.append(
                    {
                        "w_fn_ratio": round(float(w_fn), 3),
                        "lambda": round(float(lam), 3),
                        "Preprocessing": candidate_names[idx],
                        "i_emn": res["i_emn"],
                    }
                )

            sorted_scores = sorted(scores, reverse=True)
            best_idx = int(np.argmax(scores))
            winner_grid[i, j] = best_idx
            margin_grid[i, j] = sorted_scores[0] - (
                sorted_scores[1] if len(sorted_scores) > 1 else 0.0
            )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5))

    unique_winners = np.unique(winner_grid)
    remap = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_winners)}
    remapped_grid = np.vectorize(remap.get)(winner_grid)

    base_cmap = plt.get_cmap("tab10")
    colors = [base_cmap(i % 10) for i in range(len(unique_winners))]
    cmap = ListedColormap(colors)

    ax1.imshow(
        remapped_grid,
        origin="lower",
        aspect="auto",
        extent=[w_fn_values.min(), w_fn_values.max(), lambda_values.min(), lambda_values.max()],
        cmap=cmap,
        alpha=0.85,
    )
    ax1.scatter([5.0], [1.5], color="black", s=120, marker="*", zorder=5, label="Configuração Base (5.0, 1.5)")

    patches = [
        mpatches.Patch(color=colors[remap[idx]], label=candidate_names[idx])
        for idx in unique_winners
    ]
    patches.append(
        plt.Line2D(
            [0],
            [0],
            marker="*",
            color="w",
            label="Base (5.0, 1.5)",
            markerfacecolor="black",
            markersize=12,
        )
    )

    ax1.set_title(f"Regiões de Dominância (Ref. mais rápida: {ref_label})")
    ax1.set_xlabel("Severidade Clínica: Razão $w_{fn} / w_{fp}$")
    ax1.set_ylabel("Aversão ao Custo Computacional ($\\lambda$)")
    ax1.legend(handles=patches, loc="upper right", fontsize=8.5, framealpha=0.92)
    ax1.grid(alpha=0.2, linestyle="--")

    im2 = ax2.imshow(
        margin_grid,
        origin="lower",
        aspect="auto",
        extent=[w_fn_values.min(), w_fn_values.max(), lambda_values.min(), lambda_values.max()],
        cmap="viridis",
    )
    ax2.scatter([5.0], [1.5], color="white", edgecolor="black", s=120, marker="*", zorder=5)
    cbar = fig.colorbar(im2, ax=ax2)
    cbar.set_label("Diferença de $I_{EMN}$ (1º vs. 2º colocado)")

    ax2.set_title("Margem de Vitória (Robustez da Liderança)")
    ax2.set_xlabel("Severidade Clínica: Razão $w_{fn} / w_{fp}$")
    ax2.set_ylabel("Aversão ao Custo Computacional ($\\lambda$)")
    ax2.grid(alpha=0.2, linestyle="--")

    fig.suptitle(
        f"Análise de Interação 2D (Clínica × Custo) - {technique} (Ref: {ref_label})",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Diagrama de interação salvo em: {output_path}")

    return pd.DataFrame(records)


if __name__ == "__main__":
    result = calculate_marginal_efficiency()
    if result.empty:
        raise SystemExit("Nenhuma comparação válida encontrada.")
    print(result.to_string(index=False))
    sensitivity = calculate_weight_sensitivity()
    if sensitivity.empty:
        raise SystemExit("Nenhum dado válido para análise de sensibilidade.")
    plot_weight_sensitivity(sensitivity)
    plot_lambda_sensitivity(sensitivity)
    calculate_2d_interaction()