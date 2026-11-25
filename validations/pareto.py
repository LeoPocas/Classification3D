import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from scipy.optimize import curve_fit
from utils import LOGS_DIR

# --- DEFINIÇÃO DAS 3 FUNÇÕES CANDIDATAS ---
def _hyperbolic_model(t: np.ndarray, a_max: float, c: float, d: float) -> np.ndarray:
    return a_max - (c / (t + d))


def _exponential_model(t: np.ndarray, a_max: float, c: float, k: float) -> np.ndarray:
    return a_max - c * np.exp(-k * t)


def _logarithmic_model(t: np.ndarray, a: float, b: float) -> np.ndarray:
    return a + b * np.log(np.maximum(t, 1e-6))


def fit_parametric_frontier(
    pareto_df: pd.DataFrame,
    time_col: str = "train_minutes",
    acc_col: str = "accuracy",
    strict_envelope: bool = True,
) -> dict:
    """
    Ajusta as equações paramétricas sobre os pontos não dominados (Fronteira de Pareto)
    e retorna os parâmetros calibrados e a função generalizável de predição.
    """
    pts = pareto_df.sort_values(time_col).reset_index(drop=True)
    t_data = pts[time_col].to_numpy(dtype=float)
    a_data = pts[acc_col].to_numpy(dtype=float)

    if len(t_data) < 3:
        raise ValueError("São necessários pelo menos 3 pontos na fronteira para ajustar curvas de 3 parâmetros.")

    models = {}

    # 1. Ajuste Hiperbólico: A_max - c / (T + d)
    p0_hyp = [min(0.99, a_data.max() + 0.02), 0.5, 1.0]
    bounds_hyp = ([a_data.max() * 0.9, 1e-5, 1e-5], [1.0, 100.0, 100.0])
    popt_hyp, _ = curve_fit(_hyperbolic_model, t_data, a_data, p0=p0_hyp, bounds=bounds_hyp, maxfev=10000)
    pred_hyp = _hyperbolic_model(t_data, *popt_hyp)
    shift_hyp = float(np.max(a_data - pred_hyp)) if strict_envelope else 0.0
    a_max_h, c_h, d_h = popt_hyp[0] + shift_hyp, popt_hyp[1], popt_hyp[2]

    models["hyperbolic"] = {
        "rmse": float(np.sqrt(np.mean((a_data - (pred_hyp + shift_hyp)) ** 2))),
        "formula_str": f"Phi(T, A) = A - ({a_max_h:.5f} - {c_h:.5f} / (T + {d_h:.5f}))",
        "predict_frontier": lambda t, am=a_max_h, c=c_h, d=d_h: am - (c / (np.asarray(t) + d)),
    }

    # 2. Ajuste Exponencial: A_max - c * exp(-k * T)
    p0_exp = [min(0.99, a_data.max() + 0.01), 0.3, 0.15]
    bounds_exp = ([a_data.max() * 0.9, 1e-5, 1e-5], [1.0, 1.0, 5.0])
    popt_exp, _ = curve_fit(_exponential_model, t_data, a_data, p0=p0_exp, bounds=bounds_exp, maxfev=10000)
    pred_exp = _exponential_model(t_data, *popt_exp)
    shift_exp = float(np.max(a_data - pred_exp)) if strict_envelope else 0.0
    a_max_e, c_e, k_e = popt_exp[0] + shift_exp, popt_exp[1], popt_exp[2]

    models["exponential"] = {
        "rmse": float(np.sqrt(np.mean((a_data - (pred_exp + shift_exp)) ** 2))),
        "formula_str": f"Phi(T, A) = A - ({a_max_e:.5f} - {c_e:.5f} * exp(-{k_e:.5f} * T))",
        "predict_frontier": lambda t, am=a_max_e, c=c_e, k=k_e: am - c * np.exp(-k * np.asarray(t)),
    }

    # 3. Ajuste Logarítmico: a + b * ln(T)
    popt_log, _ = curve_fit(_logarithmic_model, t_data, a_data)
    pred_log = _logarithmic_model(t_data, *popt_log)
    shift_log = float(np.max(a_data - pred_log)) if strict_envelope else 0.0
    a_l, b_l = popt_log[0] + shift_log, popt_log[1]

    models["logarithmic"] = {
        "rmse": float(np.sqrt(np.mean((a_data - (pred_log + shift_log)) ** 2))),
        "formula_str": f"Phi(T, A) = A - ({a_l:.5f} + {b_l:.5f} * ln(T))",
        "predict_frontier": lambda t, a=a_l, b=b_l: a + b * np.log(np.maximum(np.asarray(t), 1e-6)),
    }

    return models


def classify_new_point(
    t_minutes: float,
    accuracy: float,
    frontier_fn,
    tolerance: float = 0.01,
) -> dict:
    """
    Classifica qualquer ponto futuro (T, A) sem precisar consultar os dados históricos.
    """
    expected_frontier_acc = float(frontier_fn(t_minutes))
    phi = accuracy - expected_frontier_acc

    if phi > 0.0:
        status = "NÃO DOMINADO (Supera a fronteira histórica)"
    elif phi >= -tolerance:
        status = "NA FRONTEIRA (Dentro da faixa eficiente)"
    else:
        status = "DOMINADO (Abaixo da eficiência esperada)"

    return {
        "T_min": t_minutes,
        "A_obs": accuracy,
        "A_fronteira_esperada": round(expected_frontier_acc, 5),
        "Phi_score": round(phi, 5),
        "Status": status,
    }


def plot_fitted_frontiers(
    pareto_df: pd.DataFrame,
    models: dict,
    all_df: pd.DataFrame | None = None,
    time_col: str = "train_minutes",
    acc_col: str = "accuracy",
    tolerance: float = 0.01,
    output_path: str = os.path.join(LOGS_DIR, "parametric_pareto_frontier.png"),
) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))

    ref_df = all_df if all_df is not None else pareto_df

    if all_df is not None:
        ax.scatter(
            all_df[time_col], all_df[acc_col],
            color="lightgray", edgecolors="gray", s=55, alpha=0.8, label="Modelos Dominados"
        )

    ax.scatter(
        pareto_df[time_col], pareto_df[acc_col],
        color="black", marker="D", s=95, zorder=5, label="Pontos Não Dominados (Pareto)"
    )

    t_grid = np.linspace(max(0.5, ref_df[time_col].min() * 0.8), ref_df[time_col].max() * 1.05, 300)

    styles = {
        "hyperbolic": ("crimson", "-", "Hiperbólica (Envoltória)"),
        "exponential": ("royalblue", "--", "Exponencial (Envoltória)"),
        "logarithmic": ("forestgreen", "-.", "Logarítmica (Envoltória)"),
    }

    for key, (color, ls, label) in styles.items():
        y_grid = models[key]["predict_frontier"](t_grid)
        ax.plot(t_grid, y_grid, color=color, linestyle=ls, linewidth=2.0, label=f"{label} (RMSE={models[key]['rmse']:.4f})")
        if key == "hyperbolic":
            ax.fill_between(
                t_grid, y_grid - tolerance, y_grid,
                color=color, alpha=0.12, label=f"Zona de Fronteira (ε = {tolerance})"
            )

    ax.set_title("Modelagem Paramétrica da Fronteira de Pareto (Generalizável para Novos Pontos)")
    ax.set_xlabel("Tempo médio de treinamento (minutos)")
    ax.set_ylabel("Acurácia média no teste")
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(loc="lower right", fontsize=8.5)
    fig.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Gráfico salvo com sucesso em: {output_path}")


# --- EXEMPLO DE USO COM OS PONTOS VISÍVEIS DO SEU GRÁFICO ---
if __name__ == "__main__":
    pareto_sample = pd.DataFrame({
        "name": ["early_ch_1", "concat_1", "early_ch_2", "early_ch_3", "concat_full"],
        "train_minutes": [2.0, 3.5, 5.3, 10.0, 72.0],
        "accuracy": [0.705, 0.750, 0.816, 0.873, 0.881],
    })

    fitted = fit_parametric_frontier(pareto_sample, strict_envelope=True)

    print("=== FÓRMULAS CALIBRADAS (PRONTAS PARA USO) ===")
    for name, info in fitted.items():
        print(f"[{name.upper()}] RMSE={info['rmse']:.5f}")
        print(f"   {info['formula_str']}\n")

    # Chamada que faltava para gerar e salvar o arquivo PNG:
    plot_fitted_frontiers(pareto_df=pareto_sample, models=fitted)

    hyp_fn = fitted["hyperbolic"]["predict_frontier"]
    novos_testes = [
        (8.0, 0.880),
        (10.0, 0.868),
        (25.0, 0.820),
    ]

    print("=== TESTE EM NOVOS MODELOS FUTUROS (SEM TABELA HISTÓRICA) ===")
    for t_new, a_new in novos_testes:
        print(classify_new_point(t_new, a_new, hyp_fn, tolerance=0.01))