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
from scipy.optimize import minimize
from scipy.stats import spearmanr
from utils import LOGS_DIR

SEGMENTED_CSV_PATH = os.path.join(LOGS_DIR, "validation_exponential_efficiency_segmented.csv")
TRAIN_SUBSET_SUFFIX = "_TrainData_30pct"

# Tolerância temporal (0.25 min = 15s) para evitar paredes verticais na fronteira por jitter de hardware
TIME_TOLERANCE_MIN = 0.25

BASE_WEIGHTS = {
    "w_fn": 2.0,
    "w_cross": 1.5,
    "w_fp": 1.0,
}


@dataclass
class ModelMetrics:
    """Métricas agregadas de um grupo experimental usando tempo de treinamento."""

    name: str
    fn: float
    fp: float
    cross_err: float
    n_nor: float
    n_hcm: float
    n_dcm: float
    accuracy: float
    train_seconds: float

    @property
    def n_total(self) -> float:
        return self.n_nor + self.n_hcm + self.n_dcm

    @property
    def train_minutes(self) -> float:
        return self.train_seconds / 60.0


def _exponential_curve(t: np.ndarray, a_max: float, c: float, k: float) -> np.ndarray:
    return a_max - c * np.exp(-k * t)


def _dataset_from_group(group_name: str) -> str:
    lower = group_name.lower()
    if lower.startswith("incor_"):
        return "incor"
    if lower.startswith("mms_"):
        return "mms"
    return "unknown"


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


def _extract_train_seconds(timing: dict) -> float:
    preferred_keys = (
        "training_seconds",
        "train_seconds",
        "training_time_seconds",
        "train_time_seconds",
        "training_time",
        "train_time",
        "fit_seconds",
    )
    for key in preferred_keys:
        val = timing.get(key)
        if isinstance(val, (int, float)) and float(val) > 0:
            return float(val)

    for key, val in timing.items():
        if "train" in key.lower() and isinstance(val, (int, float)) and float(val) > 0:
            return float(val)

    return 0.0


def _metrics_from_json(data: dict, name: str) -> Optional[ModelMetrics]:
    results = data.get("results", {})
    matrix = np.asarray(results.get("confusion_matrix", []), dtype=float)
    timing = results.get("timing", {})

    if matrix.ndim != 2 or matrix.shape[0] < 3 or matrix.shape[1] < 3:
        return None

    train_seconds = _extract_train_seconds(timing)
    n_total = float(matrix.sum())
    if train_seconds <= 0 or n_total <= 0:
        return None

    fn = float(matrix[1:, 0].sum())
    fp = float(matrix[0, 1:].sum())
    cross_err = float(matrix[1, 2] + matrix[2, 1])
    accuracy = float(np.trace(matrix) / n_total)

    n_nor = float(matrix[0, :].sum())
    n_hcm = float(matrix[1, :].sum())
    n_dcm = float(matrix[2, :].sum())

    return ModelMetrics(
        name=name,
        fn=fn,
        fp=fp,
        cross_err=cross_err,
        n_nor=n_nor,
        n_hcm=n_hcm,
        n_dcm=n_dcm,
        accuracy=accuracy,
        train_seconds=train_seconds,
    )


def load_aggregated_groups(logs_dir: str = LOGS_DIR) -> pd.DataFrame:
    print(f"[1/4] Buscando arquivos JSON em: {logs_dir}", flush=True)
    if not os.path.isdir(logs_dir):
        raise FileNotFoundError(f"Diretório de logs não encontrado: {logs_dir}")

    json_files = [
        p for p in sorted(Path(logs_dir).rglob("*.json"))
        if not p.name.startswith("analysis_")
    ]

    records: List[dict] = []
    for path in json_files:
        group_name = path.parent.name
        dataset = _dataset_from_group(group_name)
        technique = _technique_from_group(group_name)
        try:
            with path.open("r", encoding="utf-8") as stream:
                data = json.load(stream)

            metrics = _metrics_from_json(data, group_name)
            if metrics is not None:
                records.append(
                    {
                        "Dataset": dataset,
                        "Technique": technique,
                        "Group": group_name,
                        "metrics": metrics,
                    }
                )
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as err:
            print(f"      Erro ao ler {path}: {err}", flush=True)

    if not records:
        return pd.DataFrame()

    raw_df = pd.DataFrame(records)
    rows: List[dict] = []
    for (dataset, technique, group_name), gdf in raw_df.groupby(["Dataset", "Technique", "Group"]):
        ms = list(gdf["metrics"])
        rows.append(
            {
                "Dataset": dataset,
                "Train_Regime": "30pct" if technique.endswith(TRAIN_SUBSET_SUFFIX) else "100pct",
                "Technique": technique,
                "Group": group_name,
                "Preprocessing": _preprocessing_label(group_name, technique),
                "Runs": len(ms),
                "fn": float(np.mean([m.fn for m in ms])),
                "fp": float(np.mean([m.fp for m in ms])),
                "cross_err": float(np.mean([m.cross_err for m in ms])),
                "n_nor": float(np.mean([m.n_nor for m in ms])),
                "n_hcm": float(np.mean([m.n_hcm for m in ms])),
                "n_dcm": float(np.mean([m.n_dcm for m in ms])),
                "n_total": float(np.mean([m.n_total for m in ms])),
                "accuracy": float(np.mean([m.accuracy for m in ms])),
                "train_seconds": float(np.mean([m.train_seconds for m in ms])),
                "train_minutes": float(np.mean([m.train_minutes for m in ms])),
            }
        )

    print(f"[2/4] {len(rows)} grupos agregados com sucesso.", flush=True)
    return pd.DataFrame(rows)


def compute_pareto_and_dea(df: pd.DataFrame, time_tol: float = TIME_TOLERANCE_MIN) -> pd.DataFrame:
    """
    Identifica a Fronteira de Pareto com tolerância temporal (epsilon-dominância)
    para eliminar paredes verticais causadas por ruído de poucos segundos.
    """
    res = df.copy().reset_index(drop=True)
    times = res["train_minutes"].to_numpy(dtype=float)
    accs = res["accuracy"].to_numpy(dtype=float)
    n = len(res)

    dom_counts = np.zeros(n, dtype=int)
    for i in range(n):
        # j domina i se j tem acurácia maior/igual e tempo menor ou até time_tol maior
        strict_dom = (times <= times[i]) & (accs >= accs[i]) & ((times < times[i]) | (accs > accs[i]))
        eps_dom = (times <= times[i] + time_tol) & (accs > accs[i] + 1e-6)
        dom = strict_dom | eps_dom
        dom[i] = False
        dom_counts[i] = int(np.sum(dom))

    res["dominance_count"] = dom_counts
    res["is_pareto"] = res["dominance_count"] == 0

    pareto_pts = res[res["is_pareto"]].sort_values("train_minutes")
    p_times = pareto_pts["train_minutes"].to_numpy(dtype=float)
    p_accs = pareto_pts["accuracy"].to_numpy(dtype=float)

    dea = np.zeros(n, dtype=float)
    for i in range(n):
        if res.loc[i, "is_pareto"]:
            dea[i] = 1.0
        elif accs[i] <= p_accs[0]:
            dea[i] = min(1.0, p_times[0] / times[i])
        else:
            t_star = np.interp(accs[i], p_accs, p_times)
            dea[i] = min(1.0, t_star / times[i])

    res["dea_efficiency"] = np.round(dea, 5)
    return res


def fit_exponential_frontier_params(
    df_pareto: pd.DataFrame,
    w_fn: float = 2.0,
    w_cross: float = 1.5,
    w_fp: float = 1.0,
) -> Dict[str, float]:
    """
    Ajusta a envoltória exponencial via otimização restrita (SLSQP):
    minimiza a distância quadrática até os pontos de Pareto garantindo
    que A_hat(T_p) >= A_p sem precisar deslocar o teto artificialmente.
    """
    max_w = max(w_fn, w_cross, w_fp)
    pts = df_pareto[df_pareto["is_pareto"]].sort_values("train_minutes").copy()

    valid_mask = []
    for row in pts.itertuples():
        b_ref = max(0.0, 1.0 - (w_cross * row.n_dcm + w_fp * row.n_nor) / (row.n_total * max_w))
        b_cand = max(0.0, 1.0 - (w_fn * row.fn + w_cross * row.cross_err + w_fp * row.fp) / (row.n_total * max_w))
        valid_mask.append(b_cand > b_ref + 1e-6)

    valid_pts = pts[valid_mask] if any(valid_mask) else pts
    max_obs_acc = float(df_pareto["accuracy"].max())
    min_obs_acc = float(df_pareto["accuracy"].min())

    fallback = {
        "a_max": min(0.99, max_obs_acc + 0.005),
        "c": max(0.05, (max_obs_acc - min_obs_acc) + 0.02),
        "k": 0.25,
    }

    if len(valid_pts) < 2:
        return fallback

    t_data = valid_pts["train_minutes"].to_numpy(dtype=float)
    a_data = valid_pts["accuracy"].to_numpy(dtype=float)
    a_top = float(a_data.max())

    # Objetivo: minimizar soma dos quadrados dos resíduos + leve regularização para não inflar A_max
    def objective(params: np.ndarray) -> float:
        a_max, c, k = params
        pred = _exponential_curve(t_data, a_max, c, k)
        return float(np.sum((pred - a_data) ** 2) + 0.05 * (a_max - a_top) ** 2)

    # Restrição de envoltória superior: pred(t_i) - a_i >= 0 para todos os pontos da fronteira
    constraints = [
        {
            "type": "ineq",
            "fun": lambda params: _exponential_curve(t_data, params[0], params[1], params[2]) - a_data,
        }
    ]

    bounds = [
        (a_top, min(1.0, a_top + 0.015)),  # Mantém o teto colado ao platô observado
        (1e-3, 1.5),
        (1e-3, 2.0),
    ]
    x0 = np.array([min(1.0, a_top + 0.003), max(0.05, a_top - a_data.min()), 0.30])

    result = minimize(
        objective,
        x0=x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 2000, "ftol": 1e-10},
    )

    if result.success or np.all(_exponential_curve(t_data, *result.x) >= a_data - 1e-4):
        a_max_opt, c_opt, k_opt = result.x
        # Ajuste fino mínimo (apenas se houver folga numérica de 10^-4)
        residual_gap = float(np.max(a_data - _exponential_curve(t_data, a_max_opt, c_opt, k_opt)))
        a_max_final = min(1.0, float(a_max_opt + max(0.0, residual_gap)))
        return {
            "a_max": a_max_final,
            "c": float(c_opt),
            "k": float(k_opt),
        }

    return fallback


def evaluate_exponential_individual_metrics(
    df: pd.DataFrame,
    exp_params: Dict[str, float],
    w_fn: float = 2.0,
    w_cross: float = 1.5,
    w_fp: float = 1.0,
) -> pd.DataFrame:
    res = df.copy()
    max_w = max(w_fn, w_cross, w_fp)

    a_max = exp_params["a_max"]
    c = exp_params["c"]
    k = exp_params["k"]

    tau = 1.0 / k
    t_sat_95 = 3.0 * tau
    lambda_cal = math.log(2.0) / t_sat_95

    b_refs, b_cands, delta_bs = [], [], []
    i_exp_list, eta_exp_list, i_frontier_exp_list = [], [], []

    for row in res.itertuples():
        loss_ref = (w_fn * 0.0 + w_cross * row.n_dcm + w_fp * row.n_nor) / (row.n_total * max_w)
        b_ref = max(0.0, 1.0 - loss_ref)

        loss_cand = (w_fn * row.fn + w_cross * row.cross_err + w_fp * row.fp) / (row.n_total * max_w)
        b_cand = max(0.0, 1.0 - loss_cand)

        rem_margin = 1.0 - b_ref
        delta_b_rel = 0.0 if rem_margin <= 1e-9 else (b_cand - b_ref) / rem_margin
        gain_clipped = max(0.0, delta_b_rel)

        i_exp = gain_clipped * math.exp(-lambda_cal * row.train_minutes)

        if row.accuracy >= a_max - 1e-6:
            eta_exp = 1.0
        else:
            ratio = (a_max - row.accuracy) / c
            if ratio >= 1.0:
                t_star_exp = 0.0
            else:
                t_star_exp = max(0.0, -(1.0 / k) * math.log(ratio))
            eta_exp = min(1.0, t_star_exp / row.train_minutes)

        i_frontier_exp = gain_clipped * eta_exp

        b_refs.append(round(b_ref, 5))
        b_cands.append(round(b_cand, 5))
        delta_bs.append(round(delta_b_rel, 5))
        i_exp_list.append(round(i_exp, 5))
        eta_exp_list.append(round(eta_exp, 5))
        i_frontier_exp_list.append(round(i_frontier_exp, 5))

    res["b_dummy_hcm"] = b_refs
    res["b_candidate"] = b_cands
    res["delta_b_rel"] = delta_bs
    res["lambda_cal"] = round(lambda_cal, 5)
    res["i_exp"] = i_exp_list
    res["eta_exp"] = eta_exp_list
    res["i_frontier_exp"] = i_frontier_exp_list
    return res


def run_stratum_report(df: pd.DataFrame, exp_params: Dict[str, float], stratum_title: str) -> None:
    tau = 1.0 / exp_params["k"]
    pareto_pts = df[df["is_pareto"]]
    rmse_frontier = float(
        np.sqrt(
            np.mean(
                (
                    pareto_pts["accuracy"].to_numpy()
                    - _exponential_curve(
                        pareto_pts["train_minutes"].to_numpy(),
                        exp_params["a_max"],
                        exp_params["c"],
                        exp_params["k"],
                    )
                )
                ** 2
            )
        )
    )
    print("\n" + "=" * 76, flush=True)
    print(f"ESTRATO: {stratum_title}  (N = {len(df)} grupos | Pareto = {len(pareto_pts)})", flush=True)
    print("=" * 76, flush=True)
    print(
        f"Curva Exponencial: A_max={exp_params['a_max']:.5f} | "
        f"c={exp_params['c']:.5f} | k={exp_params['k']:.5f} min^-1 | "
        f"RMSE_fronteira={rmse_frontier:.5f} | Saturação 95%={3*tau:.2f} min"
    )

    if len(df) >= 3 and df["dea_efficiency"].nunique() > 1:
        rho_exp, p_exp = spearmanr(df["i_exp"], df["dea_efficiency"])
        rho_front, p_front = spearmanr(df["i_frontier_exp"], df["dea_efficiency"])
        print(
            f"Correlação Spearman com DEA -> I_exp: {rho_exp:.4f} (p={p_exp:.2e}) | "
            f"I_frontier_exp: {rho_front:.4f} (p={p_front:.2e})"
        )

    cols_preview = [
        "Technique", "Preprocessing", "train_minutes", "accuracy",
        "is_pareto", "dea_efficiency", "i_exp", "i_frontier_exp"
    ]
    print(df[cols_preview].head(10).to_string(index=False), flush=True)


def plot_stratum_dashboard(
    df: pd.DataFrame,
    exp_params: Dict[str, float],
    stratum_title: str,
    output_path: str,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(19, 6))

    # --- PAINEL 1: Acurácia x Tempo + Fronteira Exponencial ---
    ax1 = axes[0]
    vmax_color = max(0.15, float(df["i_frontier_exp"].max()))
    sc1 = ax1.scatter(
        df["train_minutes"],
        df["accuracy"],
        c=df["i_frontier_exp"],
        cmap="plasma",
        vmin=0.0,
        vmax=vmax_color,
        s=70,
        edgecolors="black",
        linewidths=0.5,
        zorder=3,
    )

    pareto_pts = df[df["is_pareto"]].sort_values("train_minutes")
    ax1.plot(
        pareto_pts["train_minutes"],
        pareto_pts["accuracy"],
        color="black",
        linestyle="-",
        linewidth=1.5,
        alpha=0.8,
        zorder=4,
        label="Fronteira Empírica (ε-Pareto)",
    )

    t_min = max(0.5, float(pareto_pts["train_minutes"].min()) * 0.95)
    t_max = max(t_min + 1.0, float(df["train_minutes"].max()) * 1.03)
    t_grid = np.linspace(t_min, t_max, 250)
    ax1.plot(
        t_grid,
        _exponential_curve(t_grid, exp_params["a_max"], exp_params["c"], exp_params["k"]),
        color="crimson",
        linestyle="--",
        linewidth=2.2,
        zorder=5,
        label="Envoltória Exponencial (SLSQP)",
    )
    plt.colorbar(sc1, ax=ax1, label="I_frontier_exp")
    ax1.set_title("1. Acurácia × Tempo (Cor = I_frontier_exp)")
    ax1.set_xlabel("Tempo médio de treinamento (min)")
    ax1.set_ylabel("Acurácia média no teste")
    ax1.legend(loc="lower right", fontsize=8.5)
    ax1.grid(alpha=0.25, linestyle="--")

    # --- PAINEL 2: Evolução dos Índices ao longo do Tempo ---
    ax2 = axes[1]
    ax2.scatter(
        df["train_minutes"],
        df["i_exp"],
        color="royalblue",
        alpha=0.85,
        s=60,
        label="I_exp (Decaimento Calibrado)",
    )
    ax2.scatter(
        df["train_minutes"],
        df["i_frontier_exp"],
        color="darkorange",
        alpha=0.85,
        marker="^",
        s=60,
        label="I_frontier_exp (Inversão Fronteira)",
    )
    t_sat = 3.0 / exp_params["k"]
    ax2.axvline(t_sat, color="gray", linestyle=":", linewidth=1.2, label=f"Saturação 95% ({t_sat:.1f} min)")

    offsets = [(7, 6), (7, -12), (7, 16)]
    top_n = df.sort_values("i_frontier_exp", ascending=False).head(min(3, len(df)))
    for rank, row in enumerate(top_n.itertuples(), start=1):
        ax2.annotate(
            f"#{rank} {row.Preprocessing}",
            xy=(row.train_minutes, row.i_frontier_exp),
            xytext=offsets[rank - 1],
            textcoords="offset points",
            fontsize=7.5,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="lightgray", alpha=0.8),
        )

    ax2.set_title("2. Comportamento dos Índices vs. Tempo")
    ax2.set_xlabel("Tempo médio de treinamento (min)")
    ax2.set_ylabel("Índice de Eficiência [0, 1]")
    ax2.set_ylim(-0.03, 1.05)
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(alpha=0.25, linestyle="--")

    # --- PAINEL 3: Validação contra a Eficiência DEA ---
    ax3 = axes[2]
    pareto_mask = df["is_pareto"]
    ax3.scatter(
        df.loc[~pareto_mask, "dea_efficiency"],
        df.loc[~pareto_mask, "i_frontier_exp"],
        color="slategray",
        alpha=0.75,
        s=60,
        label="Modelos Dominados",
    )
    ax3.scatter(
        df.loc[pareto_mask, "dea_efficiency"],
        df.loc[pareto_mask, "i_frontier_exp"],
        color="crimson",
        marker="D",
        s=90,
        edgecolors="black",
        zorder=4,
        label="Pontos na Fronteira de Pareto",
    )
    ax3.set_title("3. Validação: I_frontier_exp vs. Eficiência DEA")
    ax3.set_xlabel("Eficiência DEA (Fronteira Real do Estrato)")
    ax3.set_ylabel("I_frontier_exp (Fórmula Individual)")
    ax3.grid(alpha=0.25, linestyle="--")
    ax3.legend(loc="upper left", fontsize=8.5)

    fig.suptitle(f"Validação da Eficiência Individual — {stratum_title}", fontsize=13, fontweight="bold")
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"      Gráfico salvo em: {output_path}", flush=True)


def run_segmented_validation() -> pd.DataFrame:
    df_groups = load_aggregated_groups()
    if df_groups.empty:
        print("Nenhum dado válido encontrado nos logs.", flush=True)
        return pd.DataFrame()

    print("[3/4] Executando análise separada por Dataset e Regime de Treinamento...", flush=True)
    all_results: List[pd.DataFrame] = []

    for (dataset, regime), sub_df in df_groups.groupby(["Dataset", "Train_Regime"], sort=True):
        stratum_id = f"{dataset}_{regime}"
        stratum_title = f"Dataset: {dataset.upper()} | Treinamento: {regime}"

        df_pareto = compute_pareto_and_dea(sub_df)
        exp_params = fit_exponential_frontier_params(df_pareto, **BASE_WEIGHTS)
        df_val = evaluate_exponential_individual_metrics(df_pareto, exp_params, **BASE_WEIGHTS)

        df_val = df_val.sort_values("i_frontier_exp", ascending=False).reset_index(drop=True)
        all_results.append(df_val)

        run_stratum_report(df_val, exp_params, stratum_title)

        plot_path = os.path.join(LOGS_DIR, f"validation_{stratum_id}.png")
        plot_stratum_dashboard(df_val, exp_params, stratum_title, plot_path)

    final_df = pd.concat(all_results, ignore_index=True)
    os.makedirs(os.path.dirname(SEGMENTED_CSV_PATH), exist_ok=True)
    final_df.to_csv(SEGMENTED_CSV_PATH, index=False)
    print(f"\n[4/4] Tabela consolidada salva em: {SEGMENTED_CSV_PATH}", flush=True)
    return final_df


if __name__ == "__main__":
    run_segmented_validation()