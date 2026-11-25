import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from utils import LOGS_DIR

TRAIN_SUBSET_SUFFIX = "_TrainData_30pct"
TIME_TOLERANCE_MIN = 0.25  # 15s de tolerância para evitar degraus verticais por jitter de GPU

# ==============================================================================
# CONFIGURAÇÃO DA FÓRMULA UNIVERSAL (ESCOLHA DE IMPLEMENTAÇÃO)
# ==============================================================================
BASE_WEIGHTS = {
    "w_fn": 2.0,     # Peso para Falsos Negativos (doente classificado como normal)
    "w_cross": 1.5,  # Peso para Erro Cruzado (HCM <-> DCM)
    "w_fp": 1.0,     # Peso para Falsos Positivos (normal classificado como doente)
}

# Tempo de referência T_ref (min) onde o fator temporal 1 / (1 + T/T_ref) vale 0.50:
UNIVERSAL_STRATUM_CONFIG: Dict[Tuple[str, str], Dict[str, Optional[float]]] = {
    ("incor", "100pct"): {"t_ref_min": 25.0, "b_ref": None},
    ("incor", "30pct"): {"t_ref_min": 10.0, "b_ref": None},
    ("mms", "100pct"): {"t_ref_min": 25.0, "b_ref": None},
    ("mms", "30pct"): {"t_ref_min": 10.0, "b_ref": None},
    ("default", "default"): {"t_ref_min": 20.0, "b_ref": None},
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


def _compact_label(name: str, max_len: int = 28) -> str:
    """Abrevia prefixos longos para que o sufixo (ex: rotate_50pct) fique visível."""
    compact = (
        name.replace("CLAHE_Norm_min_max_Aug_", "CLAHE+MM+Aug_")
        .replace("CLAHE_Norm_min_max", "CLAHE+MM")
        .replace("Norm_min_max", "Norm_MM")
    )
    return compact if len(compact) <= max_len else compact[: max_len - 2] + ".."


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
    """Identifica os pontos não dominados (Fronteira e-Pareto) e Eficiência DEA para comparação."""
    res = df.copy().reset_index(drop=True)
    times = res["train_minutes"].to_numpy(dtype=float)
    accs = res["accuracy"].to_numpy(dtype=float)
    n = len(res)

    dom_counts = np.zeros(n, dtype=int)
    for i in range(n):
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


def evaluate_universal_efficiency(
    df: pd.DataFrame,
    t_ref_min: float,
    b_ref_override: Optional[float] = None,
    w_fn: float = 2.0,
    w_cross: float = 1.5,
    w_fp: float = 1.0,
) -> pd.DataFrame:
    """
    Calcula o índice adimensional com decaimento hiperbólico suave:
        I_universal = max(0, (B - B_ref) / (1 - B_ref)) * (1 / (1 + T / T_ref))
    """
    res = df.copy()
    max_w = max(w_fn, w_cross, w_fp)
    t_ref = max(1e-6, float(t_ref_min))

    b_refs, b_cands, delta_bs, time_factors, i_univ_list = [], [], [], [], []

    for row in res.itertuples():
        if b_ref_override is not None:
            b_ref = float(b_ref_override)
        else:
            loss_ref = (w_fn * 0.0 + w_cross * row.n_dcm + w_fp * row.n_nor) / (row.n_total * max_w)
            b_ref = max(0.0, 1.0 - loss_ref)

        loss_cand = (w_fn * row.fn + w_cross * row.cross_err + w_fp * row.fp) / (row.n_total * max_w)
        b_cand = max(0.0, 1.0 - loss_cand)

        rem_margin = 1.0 - b_ref
        delta_b_rel = 0.0 if rem_margin <= 1e-9 else (b_cand - b_ref) / rem_margin
        gain_clipped = max(0.0, delta_b_rel)

        # Penalidade hiperbólica adimensional (vale 0.50 quando T == T_ref)
        f_time = 1.0 / (1.0 + (row.train_minutes / t_ref))
        i_univ = gain_clipped * f_time

        b_refs.append(round(b_ref, 5))
        b_cands.append(round(b_cand, 5))
        delta_bs.append(round(delta_b_rel, 5))
        time_factors.append(round(f_time, 5))
        i_univ_list.append(round(i_univ, 5))

    res["t_ref_min"] = t_ref
    res["b_ref"] = b_refs
    res["b_candidate"] = b_cands
    res["delta_b_rel"] = delta_bs
    res["time_factor_hyp"] = time_factors
    res["i_universal"] = i_univ_list
    return res


def run_stratum_report(df: pd.DataFrame, t_ref_min: float, stratum_title: str) -> None:
    pareto_pts = df[df["is_pareto"]]
    print("\n" + "=" * 78, flush=True)
    print(
        f"ESTRATO: {stratum_title}  (N = {len(df)} grupos | Não Dominados = {len(pareto_pts)} | T_ref = {t_ref_min:.1f} min)",
        flush=True,
    )
    print("=" * 78, flush=True)

    if len(df) >= 3 and df["dea_efficiency"].nunique() > 1:
        rho_univ, p_univ = spearmanr(df["i_universal"], df["dea_efficiency"])
        print(f"Correlação Spearman (I_universal vs. DEA): rho = {rho_univ:.4f} (p = {p_univ:.2e})")

    cols_preview = [
        "Technique", "Preprocessing", "train_minutes", "accuracy",
        "is_pareto", "dea_efficiency", "delta_b_rel", "time_factor_hyp", "i_universal"
    ]
    print(df[cols_preview].head(10).to_string(index=False), flush=True)


def plot_universal_dashboard(
    df: pd.DataFrame,
    t_ref_min: float,
    stratum_title: str,
    output_path: str,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5))

    dom_df = df[~df["is_pareto"]]
    pareto_ranked = df[df["is_pareto"]].sort_values("i_universal", ascending=False).reset_index(drop=True)
    pareto_by_time = pareto_ranked.sort_values("train_minutes")

    vmax_color = max(0.15, float(df["i_universal"].max()))

    # --- PAINEL 1: Acurácia x Tempo + Reta de Iso-Eficiência + Índices Compactos ---
    ax1 = axes[0]
    sc1 = ax1.scatter(
        dom_df["train_minutes"],
        dom_df["accuracy"],
        c=dom_df["i_universal"],
        cmap="plasma",
        vmin=0.0,
        vmax=vmax_color,
        s=65,
        alpha=0.85,
        edgecolors="black",
        linewidths=0.4,
        zorder=3,
        label="Modelos Dominados",
    )

    ax1.plot(
        pareto_by_time["train_minutes"],
        pareto_by_time["accuracy"],
        color="black",
        linestyle="-",
        linewidth=1.4,
        alpha=0.75,
        zorder=4,
        label="Fronteira Empírica (ε-Pareto)",
    )

    # Limites reais de Y dos dados (impede que a curva estique o gráfico até 1.00)
    y_min_data = float(df["accuracy"].min())
    y_max_data = float(df["accuracy"].max())
    y_pad = max(0.012, (y_max_data - y_min_data) * 0.14)
    ax1.set_ylim(y_min_data - y_pad, y_max_data + y_pad)

    # Curva Implícita de Iso-Eficiência passando pelo líder (#1):
    # Com decaimento hiperbólico 1 / (1 + T/T_ref), a iso-eficiência é linear em T
    if not pareto_ranked.empty:
        leader = pareto_ranked.iloc[0]
        a_ref_est = float(df["b_ref"].iloc[0])
        i_star = float(leader["i_universal"])
        denom = (1.0 - a_ref_est) * i_star * (1.0 + float(leader["train_minutes"]) / t_ref_min)
        scale_factor = (float(leader["accuracy"]) - a_ref_est) / denom if denom > 0 else 1.0

        t_grid = np.linspace(
            max(0.2, float(df["train_minutes"].min()) * 0.8),
            float(df["train_minutes"].max()) * 1.03,
            250,
        )
        a_iso_curve = a_ref_est + scale_factor * (1.0 - a_ref_est) * i_star * (1.0 + t_grid / t_ref_min)
        ax1.plot(
            t_grid,
            a_iso_curve,
            color="crimson",
            linestyle="--",
            linewidth=1.9,
            zorder=4,
            label=f"Iso-Eficiência (I* = {i_star:.3f})",
        )

    ax1.scatter(
        pareto_ranked["train_minutes"],
        pareto_ranked["accuracy"],
        c=pareto_ranked["i_universal"],
        cmap="plasma",
        vmin=0.0,
        vmax=vmax_color,
        marker="D",
        s=135,
        edgecolors="crimson",
        linewidths=1.6,
        zorder=5,
    )

    legend_lines = ["Não Dominados (Ranking):"]
    for idx, row in pareto_ranked.iterrows():
        rank_num = idx + 1
        ax1.annotate(
            f"{rank_num}",
            xy=(row["train_minutes"], row["accuracy"]),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
            color="darkred",
            bbox=dict(boxstyle="circle,pad=0.14", fc="white", ec="crimson", lw=0.9, alpha=0.95),
            zorder=7,
        )
        short_name = _compact_label(row["Preprocessing"], max_len=28)
        legend_lines.append(f" {rank_num}. {short_name} ({row['i_universal']:.3f})")

    ax1.text(
        0.98,
        0.03,
        "\n".join(legend_lines),
        transform=ax1.transAxes,
        fontsize=7.1,
        va="bottom",
        ha="right",
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="lightgray", alpha=0.93),
        zorder=8,
    )

    ax1.axvline(t_ref_min, color="teal", linestyle=":", linewidth=1.2, alpha=0.8, label=f"T_ref = {t_ref_min:.0f} min")
    plt.colorbar(sc1, ax=ax1, label="I_universal")
    ax1.set_title("1. Acurácia × Tempo + Iso-Eficiência")
    ax1.set_xlabel("Tempo médio de treinamento (min)")
    ax1.set_ylabel("Acurácia média no teste")
    ax1.legend(loc="upper left", fontsize=7.6)
    ax1.grid(alpha=0.25, linestyle="--")

    # --- PAINEL 2: I_universal vs. Tempo + Isolinhas de Ganho Clínico ---
    ax2 = axes[1]
    t_grid_2 = np.linspace(0.1, max(float(df["train_minutes"].max()) * 1.05, t_ref_min * 1.5), 250)
    for delta_b_level, ls in [(1.0, "--"), (0.75, ":"), (0.50, "-.")]:
        ax2.plot(
            t_grid_2,
            delta_b_level / (1.0 + t_grid_2 / t_ref_min),
            color="gray",
            linestyle=ls,
            linewidth=1.0,
            alpha=0.55,
            label=f"Isolinha ΔB_rel = {delta_b_level:.2f}",
        )

    ax2.scatter(
        dom_df["train_minutes"],
        dom_df["i_universal"],
        color="royalblue",
        alpha=0.75,
        s=60,
        zorder=3,
        label="Modelos Dominados",
    )
    ax2.scatter(
        pareto_ranked["train_minutes"],
        pareto_ranked["i_universal"],
        color="crimson",
        marker="D",
        s=105,
        edgecolors="black",
        linewidths=1.0,
        zorder=5,
        label="Pontos Não Dominados",
    )
    for idx, row in pareto_ranked.iterrows():
        ax2.annotate(
            f"{idx + 1}",
            xy=(row["train_minutes"], row["i_universal"]),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            fontweight="bold",
            color="darkred",
        )

    ax2.axvline(t_ref_min, color="teal", linestyle=":", linewidth=1.2, alpha=0.8)
    ax2.set_title("2. Índice Universal vs. Tempo (Fórmula Fechada)")
    ax2.set_xlabel("Tempo médio de treinamento (min)")
    ax2.set_ylabel("I_universal [0, 1]")
    ax2.set_ylim(-0.04, 1.05)
    ax2.legend(loc="upper right", fontsize=7.8)
    ax2.grid(alpha=0.25, linestyle="--")

    # --- PAINEL 3: Validação Independente contra DEA ---
    ax3 = axes[2]
    ax3.scatter(
        dom_df["dea_efficiency"],
        dom_df["i_universal"],
        color="slategray",
        alpha=0.75,
        s=60,
        zorder=3,
        label="Modelos Dominados",
    )
    ax3.scatter(
        pareto_ranked["dea_efficiency"],
        pareto_ranked["i_universal"],
        color="crimson",
        marker="D",
        s=105,
        edgecolors="black",
        linewidths=1.0,
        zorder=5,
        label="Não Dominados (DEA = 1.0)",
    )
    for idx, row in pareto_ranked.iterrows():
        ax3.annotate(
            f"{idx + 1}",
            xy=(row["dea_efficiency"], row["i_universal"]),
            xytext=(7, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            fontweight="bold",
            color="darkred",
        )

    ax3.set_title("3. Validação: I_universal vs. Eficiência DEA")
    ax3.set_xlabel("Eficiência DEA (Fronteira Empírica)")
    ax3.set_ylabel("I_universal (Sem Histórico)")
    ax3.set_xlim(0.05, 1.12)
    ax3.grid(alpha=0.25, linestyle="--")
    ax3.legend(loc="upper left", fontsize=8)

    fig.suptitle(
        f"Índice Universal Adimensional (T_ref = {t_ref_min:.0f} min) — {stratum_title}",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"      Gráfico salvo em: {output_path}", flush=True)


def run_universal_validation() -> pd.DataFrame:
    df_groups = load_aggregated_groups()
    if df_groups.empty:
        print("Nenhum dado válido encontrado nos logs.", flush=True)
        return pd.DataFrame()

    print("[3/4] Avaliando Índice Universal por Dataset e Regime...", flush=True)
    all_results: List[pd.DataFrame] = []

    for (dataset, regime), sub_df in df_groups.groupby(["Dataset", "Train_Regime"], sort=True):
        stratum_id = f"{dataset}_{regime}"
        stratum_title = f"Dataset: {dataset.upper()} | Treinamento: {regime}"

        cfg = UNIVERSAL_STRATUM_CONFIG.get(
            (dataset, regime),
            UNIVERSAL_STRATUM_CONFIG[("default", "default")],
        )
        t_ref_min = float(cfg["t_ref_min"])
        b_ref_override = cfg.get("b_ref")

        df_pareto = compute_pareto_and_dea(sub_df)
        df_val = evaluate_universal_efficiency(
            df_pareto,
            t_ref_min=t_ref_min,
            b_ref_override=b_ref_override,
            **BASE_WEIGHTS,
        )
        df_val = df_val.sort_values("i_universal", ascending=False).reset_index(drop=True)
        all_results.append(df_val)

        run_stratum_report(df_val, t_ref_min, stratum_title)

        plot_path = os.path.join(LOGS_DIR, f"validation_universal_{stratum_id}.png")
        plot_universal_dashboard(df_val, t_ref_min, stratum_title, plot_path)

    final_df = pd.concat(all_results, ignore_index=True)
    csv_path = os.path.join(LOGS_DIR, "validation_universal_efficiency.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    final_df.to_csv(csv_path, index=False)
    print(f"\n[4/4] Tabela consolidada salva em: {csv_path}", flush=True)
    return final_df


if __name__ == "__main__":
    run_universal_validation()