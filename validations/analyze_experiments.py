import os
import json
import numpy as np
import pandas as pd
from datetime import timedelta
from pathlib import Path
from Classification3D.utils import LOGS_DIR

def load_logs():
    """Lê todos os JSONs nas subpastas de 'tempos' e organiza os dados."""
    experiments_data = []

    if not os.path.exists(LOGS_DIR):
        print(f"Diretório de logs não encontrado: {LOGS_DIR}")
        return pd.DataFrame()

    log_root = Path(LOGS_DIR)
    # O formato atual é tempos/<dataset>/<modelo>/<grupo>/<arquivo>.json.
    for file_path in sorted(log_root.rglob("*.json")):
        if file_path.name.startswith("analysis_"):
            continue

        relative_parts = file_path.relative_to(log_root).parts
        if len(relative_parts) < 4:
            continue

        dataset, model, group_name = relative_parts[:3]
        json_file = file_path.name

        try:
            with file_path.open("r", encoding="utf-8") as file:
                data = json.load(file)

            results = data.get("results", {})
            timing = results.get("timing", {})

            # Se não tiver timing (logs antigos), ignoramos ou colocamos NaN
            if not timing:
                continue

            raw_test_results = results.get("test_results_raw")
            test_loss = (
                raw_test_results[0]
                if isinstance(raw_test_results, list) and raw_test_results
                else np.nan
            )

            experiments_data.append({
                "Dataset": dataset,
                "Model": model,
                "Group": group_name,
                "File": json_file,
                "Load Time (s)": timing.get("load_data_seconds", np.nan),
                "Train Time (s)": timing.get("training_seconds", np.nan),
                "Test Time (s)": timing.get("testing_seconds", np.nan),
                "Test Accuracy": results.get("test_accuracy", np.nan),
                "Test AUC": results.get("test_auc", np.nan),
                "Test Loss": test_loss
            })
        except Exception as e:
            print(f"Erro ao ler {file_path}: {e}")

    return pd.DataFrame(experiments_data)

def analyze_and_report():
    df = load_logs()
    
    if df.empty:
        print("Nenhum log com dados de tempo encontrado.")
        return

    # Agrupa pelo caminho lógico completo para não misturar datasets/técnicas.
    analyze_columns = [
        "Load Time (s)", "Train Time (s)", "Test Time (s)", 
        "Test Accuracy", "Test AUC", "Test Loss"
    ]
    
    # Filtra colunas que realmente existem no DF (para evitar erro se todos forem NaN)
    analyze_columns = [col for col in analyze_columns if col in df.columns]

    print("\n" + "="*80)
    print(f"{'ANÁLISE ESTATÍSTICA DE EXECUÇÃO E RESULTADOS':^80}")
    print("="*80)

    # Convertendo segundos para timedelta para visualização textual bonita é complicado em stats agregados,
    # então manteremos segundos e minutos nos prints.

    group_columns = ["Dataset", "Model", "Group"]
    stats_df = df.groupby(group_columns)[analyze_columns].agg(
        ['count', 'mean', 'std', 'min', 'max']
    )
    stats_df.columns = [f"{column}_{statistic}" for column, statistic in stats_df.columns]
    stats_df.index.names = group_columns
    
    # Exibir relatório textual
    for group in stats_df.index:
        dataset, model, group_name = group
        print(f"\nGrupo: {dataset}/{model}/{group_name}")
        print("-" * 40)
        n_samples = len(
            df[
                (df["Dataset"] == dataset)
                & (df["Model"] == model)
                & (df["Group"] == group_name)
            ]
        )
        print(f"Execuções válidas: {int(n_samples)}")
        
        for col in analyze_columns:
            mean_val = stats_df.loc[group, f"{col}_mean"]
            std_val = stats_df.loc[group, f"{col}_std"]
            min_val = stats_df.loc[group, f"{col}_min"]
            max_val = stats_df.loc[group, f"{col}_max"]
            
            # Coeficiente de Variação (CV) = (Desvio Padrão / Média) * 100
            cv = (std_val / mean_val * 100) if mean_val > 0 else 0.0
            
            # Formatação diferente para Tempo vs Métricas (0-1)
            if "Time" in col:
                unit = "s"
                val_fmt = "{:8.2f}"
            else:
                unit = ""
                val_fmt = "{:8.4f}"

            print(f"  > {col}:")
            print(f"    Média:   {val_fmt.format(mean_val)}{unit}")
            if "Time" in col:
                 print(f"             ({timedelta(seconds=int(mean_val))})")
            print(f"    Desvio:  {val_fmt.format(std_val)}{unit}")
            print(f"    CV (%):  {cv:8.2f}%")
            print(f"    Min/Max: {val_fmt.format(min_val)} - {val_fmt.format(max_val)}")

    # Análise extra: Salvar em CSV
    output_csv = os.path.join(LOGS_DIR, "analysis_summary.csv")
    stats_df.to_csv(output_csv)
    print("\n" + "="*80)
    print(f"Relatório resumido salvo em: {output_csv}")
    print("="*80)

if __name__ == "__main__":
    analyze_and_report()
