import os
import json
import numpy as np
import pandas as pd
from datetime import timedelta
from Classification3D.utils import LOGS_DIR

def load_logs():
    """Lê todos os JSONs nas subpastas de 'tempos' e organiza os dados."""
    experiments_data = []

    if not os.path.exists(LOGS_DIR):
        print(f"Diretório de logs não encontrado: {LOGS_DIR}")
        return []

    # Percorre todas as pastas de grupos (ex: Baseline_Raw, CLAHE, etc.)
    for group_name in os.listdir(LOGS_DIR):
        group_path = os.path.join(LOGS_DIR, group_name)
        
        if not os.path.isdir(group_path):
            continue
            
        json_files = [f for f in os.listdir(group_path) if f.endswith('.json') and not f.startswith('analysis_')]
        
        for json_file in json_files:
            file_path = os.path.join(group_path, json_file)
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                results = data.get("results", {})
                timing = results.get("timing", {})
                
                # Se não tiver timing (logs antigos), ignoramos ou colocamos NaN
                if not timing:
                    continue

                experiments_data.append({
                    "Group": group_name,
                    "File": json_file,
                    "Load Time (s)": timing.get("load_data_seconds", np.nan),
                    "Train Time (s)": timing.get("training_seconds", np.nan),
                    "Test Time (s)": timing.get("testing_seconds", np.nan),
                    "Test Accuracy": results.get("test_accuracy", np.nan),
                    "Test AUC": results.get("test_auc", np.nan),
                    # Tenta pegar test loss da lista raw (index 0) ou de alguma chave especifica
                    "Test Loss": results.get("test_results_raw", [np.nan])[0] if isinstance(results.get("test_results_raw"), list) else np.nan
                })
            except Exception as e:
                print(f"Erro ao ler {file_path}: {e}")

    return pd.DataFrame(experiments_data)

def analyze_and_report():
    df = load_logs()
    
    if df.empty:
        print("Nenhum log com dados de tempo encontrado.")
        return

    # Agrupa por 'Group' e calcula estatísticas para as colunas
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

    stats_df = df.groupby("Group")[analyze_columns].agg(['count', 'mean', 'std', 'min', 'max'])
    
    # Exibir relatório textual
    for group in stats_df.index:
        print(f"\nGrupo: {group}")
        print("-" * 40)
        n_samples = stats_df.loc[group, (analyze_columns[0], 'count')]
        print(f"Execuções válidas: {int(n_samples)}")
        
        for col in analyze_columns:
            mean_val = stats_df.loc[group, (col, 'mean')]
            std_val = stats_df.loc[group, (col, 'std')]
            min_val = stats_df.loc[group, (col, 'min')]
            max_val = stats_df.loc[group, (col, 'max')]
            
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
