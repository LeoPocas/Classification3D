import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from utils import LOGS_DIR, OUTPUT_PATH


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
    """Calcula o Índice de Eficiência Marginal Normalizada (I_EMN)."""

    def __init__(
        self,
        w_fn: float = 5.0,
        w_cross: float = 2.0,
        w_fp: float = 1.0,
        lambda_param: float = 0.5,
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
        self, baseline: ModelMetrics, candidate: ModelMetrics
    ) -> Dict[str, float]:
        b_base = self.compute_clinical_benefit(baseline)
        b_cand = self.compute_clinical_benefit(candidate)
        remaining_margin = 1.0 - b_base
        delta_b_rel = (
            0.0
            if remaining_margin <= 1e-9
            else (b_cand - b_base) / remaining_margin
        )

        if baseline.cost_seconds <= 0 or candidate.cost_seconds <= 0:
            raise ValueError("Os custos computacionais devem ser positivos.")

        cost_ratio = candidate.cost_seconds / baseline.cost_seconds
        cost_penalty = max(
            0.001, 1.0 + self.lambda_param * math.log2(cost_ratio)
        )

        return {
            "model_name": candidate.name,
            "b_baseline": round(b_base, 5),
            "b_candidate": round(b_cand, 5),
            "delta_b_rel": round(delta_b_rel, 5),
            "cost_ratio": round(cost_ratio, 5),
            "cost_penalty": round(cost_penalty, 5),
            "i_emn": round(delta_b_rel / cost_penalty, 5),
        }


def _technique_from_group(group_name: str) -> str:
    """Extrai a técnica base removendo o sufixo da configuração."""
    suffix_markers = (
        "_Baseline_Raw",
        "_CLAHE",
        "_Norm_min_max",
        "_ROI",
        "_Aug_rotate",
    )
    positions = [
        group_name.find(marker)
        for marker in suffix_markers
        if group_name.find(marker) >= 0
    ]
    return group_name[: min(positions)] if positions else group_name


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

    for group_name in sorted(os.listdir(logs_dir)):
        if "mms" in group_name.lower():
            continue
        group_path = os.path.join(logs_dir, group_name)
        if not os.path.isdir(group_path):
            continue

        technique = _technique_from_group(group_name)
        files = sorted(
            f
            for f in os.listdir(group_path)
            if f.endswith(".json") and not f.startswith("analysis_")
        )

        for filename in files:
            path = os.path.join(group_path, filename)
            try:
                with open(path, "r", encoding="utf-8") as stream:
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


def calculate_marginal_efficiency(
    baseline_suffix: str = "Baseline_Raw",
    logs_dir: str = LOGS_DIR,
    output_path: str = os.path.join(
        OUTPUT_PATH, "marginal_efficiency_summary.csv"
    ),
    evaluator: Optional[MarginalEfficiencyEvaluator] = None,
) -> pd.DataFrame:
    """Avalia cada grupo contra o baseline global e único da sua técnica."""
    evaluator = evaluator or MarginalEfficiencyEvaluator()
    df = load_metrics_dataframe(logs_dir)
    if df.empty:
        print("Nenhum dado válido extraído.")
        return pd.DataFrame()

    rows: List[dict] = []

    for technique, tech_df in df.groupby("Technique"):
        baseline_group_name = f"{technique}_{baseline_suffix}"
        baseline_runs = tech_df[tech_df["Group"] == baseline_group_name]

        if baseline_runs.empty:
            print(
                f"Aviso: Baseline '{baseline_group_name}' não encontrado para técnica '{technique}'."
            )
            continue

        # 1. BASELINE ÚNICO E GLOBAL DA TÉCNICA
        base_metrics_list = [row.metrics for row in baseline_runs.itertuples()]
        global_baseline = ModelMetrics(
            baseline_group_name,
            np.mean([m.fn for m in base_metrics_list]),
            np.mean([m.fp for m in base_metrics_list]),
            np.mean([m.cross_err for m in base_metrics_list]),
            np.mean([m.n_total for m in base_metrics_list]),
            np.mean([m.cost_seconds for m in base_metrics_list]),
        )

        # 2. AVALIAÇÃO DOS CANDIDATOS DA TÉCNICA
        candidate_groups = [
            g for g in tech_df["Group"].unique() if g != baseline_group_name
        ]

        for group_name in candidate_groups:
            candidate_runs = tech_df[tech_df["Group"] == group_name]
            if candidate_runs.empty:
                continue

            cand_metrics_list = [
                row.metrics for row in candidate_runs.itertuples()
            ]
            candidate = ModelMetrics(
                group_name,
                np.mean([m.fn for m in cand_metrics_list]),
                np.mean([m.fp for m in cand_metrics_list]),
                np.mean([m.cross_err for m in cand_metrics_list]),
                np.mean([m.n_total for m in cand_metrics_list]),
                np.mean([m.cost_seconds for m in cand_metrics_list]),
            )

            # Compara o candidato contra o baseline único da técnica
            result = evaluator.evaluate_candidate(global_baseline, candidate)

            rows.append(
                {
                    "Technique": technique,
                    "Group": group_name,
                    "Baseline_Runs": len(baseline_runs),
                    "Candidate_Runs": len(candidate_runs),
                    **result,
                }
            )

    summary = pd.DataFrame(rows)
    if not summary.empty:
        # Ordena por Técnica (A-Z) e depois por i_emn (maior para o menor)
        summary = summary.sort_values(
            ["Technique", "i_emn"], ascending=[True, False]
        ).reset_index(drop=True)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        summary.to_csv(output_path, index=False)
        print(f"Resumo salvo em: {output_path}")

    return summary


if __name__ == "__main__":
    result = calculate_marginal_efficiency()
    if result.empty:
        raise SystemExit("Nenhuma comparação válida encontrada.")
    print(result.to_string(index=False))