import os
import time
import json
import datetime
import traceback
# Importando do novo arquivo Runner em vez do Training original (que não pudemos alterar)
from Classification3D.models.incor.incorDualRunner import run_incor_dual_training
from Classification3D.models.incor.incorConcatRunner import run_incor_concat_training

from Classification3D.utils import OUTPUT_PATH

# ==========================================
# CONFIGURAÇÃO DA EXECUÇÃO (CONTROLE CENTRAL)
# ==========================================
EXPERIMENT_CONFIG = {
    # "experiment_name": "Incor_Dual_Baseline", 
    # "description": "Execução base para estabelecer linha de base de tempo e métricas.",
    "model_mode": "concat", # 'dual' ou 'concat'
    "experiment_name": "Incor_Concat_Baseline",
    "description": "Execução com volumes sístole/diástole concatenados em profundidade (24 fatias).",

    # Hiperparâmetros de Treino
    "epochs"                    : 300,
    "batch_size"                : 8, 
    "predictions_batch_size"    : 4,
    "learning_rate"             : 0.0001,
    "test_split"                : 0.10,
    "random_seed"               : 41,
    
    # Flags de Pré-processamento (Estrutura pronta para implementação futura nos loaders)
    "preprocessing": {
        "apply_roi"             : False,
        "apply_clahe"           : True,
        "normalization"         : "min_max", #"min_max", ou 'z_score', None
        "resampling"            : False,
        "augmentation"          : 'rotate', #'zoom', 'rotate+zoom', or None        
        "augmentation_rate"     : 1.0, # 0.0 a 1.0 (Porcentagem do dataset que sofrerá augmentation)        
        "save_debug_images"     : False
    }, 
    
    # Controle de Outputs
    "save_weights": True,
    "weights_filename_loss": "incorMax2_loss.weights.keras",
    "weights_filename_auc": "incorMax2_auc.weights.keras"
}  

def get_experiment_group_folder(config):
    """
    Gera um nome de pasta baseado nas flags de pré-processamento ativas.
    """
    mode_prefix = config.get("model_mode", "unknown") + '_'
    preprocessing_config = config.get("preprocessing", {})

    active_flags = []
    
    if preprocessing_config.get("apply_roi"):
        active_flags.append("ROI")
        
    if preprocessing_config.get("apply_clahe"):
        active_flags.append("CLAHE")
        
    norm = preprocessing_config.get("normalization")
    if norm:
        active_flags.append(f"Norm_{norm}")
        
    if preprocessing_config.get("resampling"):
        active_flags.append("Resampled")
        
    aug = preprocessing_config.get("augmentation")
    if aug:
        rate = preprocessing_config.get("augmentation_rate", 1.0)
        # Formatar 0.5 como '50pct' ou '0.5'
        rate_str = f"{int(rate*100)}pct"
        active_flags.append(f"Aug_{aug}_{rate_str}")

    # Se nenhuma flag relevante estiver ativa
    if not active_flags:
        return mode_prefix + "Baseline_Raw"
        
    return mode_prefix + "_".join(active_flags)

def save_execution_log(config, results, duration_str, timestamp):
    """Salva os dados da execução na pasta 'tempos', organizada por tipo de experimento."""
    
    # Determina o subgrupo baseando-se no pre-processamento
    group_folder = get_experiment_group_folder(config)
    
    # Caminho: tempos/Nome_Do_Grupo/
    log_dir = os.path.join(os.path.dirname(__file__), "tempos", group_folder)
    os.makedirs(log_dir, exist_ok=True)
    
    filename = f"exp_{timestamp}_{config['experiment_name']}.json"
    filepath = os.path.join(log_dir, filename)
    
    log_data = {
        "timestamp": timestamp,
        "duration": duration_str,
        "group": group_folder,
        "config": config,
        "results": results
    }
    
    with open(filepath, 'w') as f:
        json.dump(log_data, f, indent=4, default=str)
    
    print(f"\n[LOG] Relatório de execução salvo em: {filepath}")

def main():
    print(f"=== Iniciando Experimento: {EXPERIMENT_CONFIG['experiment_name']} ===")
    start_time = time.time()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    try:
        if EXPERIMENT_CONFIG.get("model_mode") == "concat":
            training_results = run_incor_concat_training(EXPERIMENT_CONFIG)
        else:
            training_results = run_incor_dual_training(EXPERIMENT_CONFIG)
        
        status = "SUCCESS"
    except Exception as e:
        print(f"!!! Erro durante a execução: {str(e)}")
        traceback.print_exc()
        training_results = {"error": str(e), "traceback": traceback.format_exc()}
        status = "FAILED"

    end_time = time.time()
    duration_seconds = end_time - start_time
    duration_str = str(datetime.timedelta(seconds=int(duration_seconds)))
    
    print(f"\n=== Experimento Finalizado ({status}) ===")
    print(f"Duração Total (Script): {duration_str}")
    
    if status == "SUCCESS" and "timing" in training_results:
        timing = training_results["timing"]
        load_sec = timing.get("load_data_seconds", 0)
        train_sec = timing.get("training_seconds", 0)
        test_sec = timing.get("testing_seconds", 0)
        
        load_str = str(datetime.timedelta(seconds=int(load_sec)))
        train_str = str(datetime.timedelta(seconds=int(train_sec)))
        test_str = str(datetime.timedelta(seconds=int(test_sec)))
        
        print(f"Tempo Carregamento Dados (Treino): {load_str} ({load_sec:.2f}s)")
        print(f"Tempo de Treinamento Puro:         {train_str} ({train_sec:.2f}s)")
        print(f"Tempo de Teste e Inferência:       {test_str} ({test_sec:.2f}s)")
    
    # Salvar logs
    save_execution_log(EXPERIMENT_CONFIG, training_results, duration_str, timestamp)

if __name__ == "__main__":
    main()
