import os
import json
import numpy as np
import pandas as pd
from utils import LOGS_DIR

def calculate_relative_efficiency(baseline_name="Baseline_Raw", metric_perf="test_f1"):
    """
    Calcula o Ganho Relativo de Desempenho, Custo Relativo de Tempo
    e a Eficiência Relativa para cada pipeline em relação ao Baseline.
    """
    if not os.path.exists(LOGS_DIR):
        print(f"Diretório de logs não encontrado: {LOGS_DIR}")
        return

    records = []

    for group_name in sorted(os.listdir(LOGS_DIR)):
        if group_name.startswith("early") or group_name.startswith("concat"):
            continue

        group_path = os.path.join(LOGS_DIR, group_name)
        if not os.path.isdir(group_path):
            continue

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

                # Tenta buscar a métrica solicitada ou fallback para acurácia
                perf_val = results.get(metric_perf)
                if perf_val is None and metric_perf == "test_f1":
                    perf_val = results.get("test_f1_macro", results.get("test_accuracy", np.nan))

                record = {
                    "Group": group_name,
                    "Run_Order": execution_idx,
                    "perf_metric": perf_val,
                    "training_seconds": timing.get("training_seconds", np.nan),
                }
                records.append(record)
            except Exception as e:
                print(f"Erro ao ler {file_path}: {e}")

    df = pd.DataFrame(records)

    if df.empty:
        print("Nenhum dado válido extraído.")
        return

    # Agrupa por técnica e calcula as médias das métricas
    summary = df.groupby("Group").agg(
        mean_perf=("perf_metric", "mean"),
        mean_time=("training_seconds", "mean"),
        n_samples=("perf_metric", "count")
    ).reset_index()

    if baseline_name not in summary["Group"].values:
        print(f"Baseline '{baseline_name}' não encontrado nos dados.")
        return

    # Extrai os valores médios do Baseline
    baseline_row = summary[summary["Group"] == baseline_name].iloc[0]
    base_perf = baseline_row["mean_perf"]
    base_time = baseline_row["mean_time"]

    # Cálculo das métricas relativas
    summary["Ganho_Relativo_Desempenho"] = summary["mean_perf"] / base_perf
    summary["Custo_Relativo_Tempo"] = summary["mean_time"] / base_time
    summary["Eficiencia_Relativa"] = (
        summary["Ganho_Relativo_Desempenho"] / summary["Custo_Relativo_Tempo"]
    )

    # Ordena por Eficiência Relativa decrescente
    summary = summary.sort_values(by="Eficiencia_Relativa", ascending=False)

    print("\n" + "=" * 110)
    print(f"{'ANÁLISE DE EFICIÊNCIA RELATIVA (BASELINE: ' + baseline_name + ')':^110}")
    print("=" * 110)
    print(f"{'Grupo Comparado':<38} | {'F1 / Métrica':<12} | {'Tempo (s)':<10} | {'Ganho Rel.':<11} | {'Custo Rel.':<11} | {'Eficiência'}")
    print("-" * 110)

    for _, row in summary.iterrows():
        is_baseline = " (Baseline)" if row["Group"] == baseline_name else ""
        print(
            f"{row['Group'] + is_baseline:<38} | "
            f"{row['mean_perf']:<12.4f} | "
            f"{row['mean_time']:<10.1f} | "
            f"{row['Ganho_Relativo_Desempenho']:<11.4f} | "
            f"{row['Custo_Relativo_Tempo']:<11.4f} | "
            f"{row['Eficiencia_Relativa']:.4f}"
        )

    print("=" * 110 + "\n")

    return summary

if __name__ == "__main__":
    calculate_relative_efficiency(baseline_name="Baseline_Raw", metric_perf="test_f1")