import os
import json
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from utils import LOGS_DIR

def load_logs_dataframe():
    """
    Carrega os arquivos JSON organizando as repetições sequenciais de cada técnica
    para viabilizar o pareamento 1:1 contra o Baseline.
    """
    if not os.path.exists(LOGS_DIR):
        print(f"Diretório de logs não encontrado: {LOGS_DIR}")
        return pd.DataFrame()

    records = []

    for group_name in sorted(os.listdir(LOGS_DIR)):
        # Ignora filtros específicos
        if group_name.startswith("early") or group_name.startswith("concat"):
            continue

        group_path = os.path.join(LOGS_DIR, group_name)
        if not os.path.isdir(group_path):
            continue

        # Ordena arquivos para garantir alinhamento ordinal das repetições
        json_files = sorted([
            f for f in os.listdir(group_path) 
            if f.endswith('.json') and not f.startswith('analysis_')
        ])

        for execution_idx, json_file in enumerate(json_files):
            file_path = os.path.join(group_path, json_file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                results = data.get("results", {})
                timing = results.get("timing", {})

                record = {
                    "Group": group_name,
                    "Run_Order": execution_idx,  # Índice de pareamento por ordem de repetição (0, 1, 2...)
                    "File": json_file,
                    "test_accuracy": results.get("test_accuracy", np.nan),
                    "test_auc": results.get("test_auc", np.nan),
                    "training_seconds": timing.get("training_seconds", np.nan),
                }
                records.append(record)
            except Exception as e:
                print(f"Erro ao ler {file_path}: {e}")

    return pd.DataFrame(records)


def run_wilcoxon_paired_scenario_a(baseline_name="Baseline_Raw", alpha_global=0.05):
    """
    Executa o Teste de Wilcoxon pareado por repetição (Cenário A),
    calculando Bonferroni sobre as K comparações válidas contra o Baseline.
    """
    df = load_logs_dataframe()

    if df.empty:
        print("Nenhum dado encontrado nos logs.")
        return

    # Identifica os grupos a comparar
    groups = [g for g in df["Group"].unique() if g != baseline_name]
    K = len(groups)

    if K == 0:
        print(f"Nenhum grupo diferente do baseline '{baseline_name}' foi encontrado.")
        return

    # Correção de Bonferroni
    alpha_bonferroni = alpha_global / K

    print("\n" + "=" * 100)
    print(f"{'TESTE DE WILCOXON PAREADO COM BONFERRONI (CENÁRIO A - PAREAMENTO POR REPETIÇÃO)':^100}")
    print("=" * 100)
    print(f"Baseline: '{baseline_name}' | Total de Técnicas Comparadas (K): {K}")
    print(f"Significância Global (α): {alpha_global} | Significância Ajustada (α_ajustado = α/K): {alpha_bonferroni:.6f}\n")

    metrics_to_test = [
        ("Acurácia de Teste", "test_accuracy"),
        ("Tempo de Treinamento (s)", "training_seconds")
    ]

    # Extrai o subset do Baseline para realizar os merges par a par
    baseline_df = df[df["Group"] == baseline_name]

    for metric_label, metric_col in metrics_to_test:
        print("-" * 100)
        print(f" MÉTRICA ANALISADA: {metric_label.upper()}")
        print("-" * 100)
        print(f"{'Grupo Comparado':<38} | {'Pares (N)':<9} | {'p-valor Bruto':<13} | {'p-val (Bonf.)':<13} | {'Significativo?'}")
        print("-" * 100)

        base_sub = baseline_df[["Run_Order", metric_col]].rename(columns={metric_col: "baseline_val"}).dropna()

        for group in groups:
            group_sub = df[df["Group"] == group][["Run_Order", metric_col]].rename(columns={metric_col: "model_val"}).dropna()

            # Merge par a par mantendo apenas as repetições correspondentes
            merged = pd.merge(base_sub, group_sub, on="Run_Order").dropna()
            n_pairs = len(merged)

            if n_pairs < 6:
                print(f"{group:<38} | {n_pairs:<9} | Insuficiente (Requer N >= 6 para estatística mínima)")
                continue

            b_vals = merged["baseline_val"].values
            m_vals = merged["model_val"].values

            # Tratamento caso todas as diferenças sejam nulas
            diffs = m_vals - b_vals
            if np.all(diffs == 0):
                print(f"{group:<38} | {n_pairs:<9} | {'0.0':<13} | {'1.0':<13} | Não (Valores Idênticos)")
                continue

            try:
                # Teste de Wilcoxon Pareado (Signed-Rank)
                stat, p_value = wilcoxon(b_vals, m_vals, zero_method='pratt')
                
                # Ajuste de Bonferroni (p * K)
                p_value_bonf = min(p_value * K, 1.0)
                
                # A decisão de significância é baseada na comparação p_bruto < alpha_ajustado (ou p_bonf < alpha_global)
                is_sig = p_value < alpha_bonferroni
                sig_str = "SIM (*)" if is_sig else "Não"

                print(f"{group:<38} | {n_pairs:<9} | {p_value:<13.5e} | {p_value_bonf:<13.5e} | {sig_str}")

            except Exception as e:
                print(f"{group:<38} | {n_pairs:<9} | Erro: {e}")

    print("=" * 100 + "\n")


if __name__ == "__main__":
    run_wilcoxon_paired_scenario_a(baseline_name="Baseline_Raw", alpha_global=0.01)