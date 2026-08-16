import os
import sys
import time
import json
import datetime
import traceback
import subprocess
import argparse


# 1. FORÇA O KERAS/TENSORFLOW A ALOCAR VRAM DINAMICAMENTE (NÃO TUDO DE UMA VEZ)
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

# ==========================================
# CONFIGURAÇÃO DA EXECUÇÃO (CONTROLE CENTRAL)
# ==========================================
EXPERIMENT_CONFIG = {
    "model_mode": "concat_volume", #"concat_volume", # 'dual' ou 'early_channel' 
    "experiment_name": "Incor_Channel",
    "description": "Execução com volumes sístole/diástole concatenados por canais.",

    # Hiperparâmetros de Treino
    "epochs"                    : 300,
    "batch_size"                : 8, 
    "predictions_batch_size"    : 4,
    "learning_rate"             : 0.0001,
    "test_split"                : 0.10,
    "random_seed"               : 41,
    
    # Flags de Pré-processamento
    "preprocessing": {
        "apply_roi"             : False,
        "apply_clahe"           : True,
        "normalization"         : "min_max", #"min_max", ou 'z_score', None
        "resampling"            : False,
        "augmentation"          : 'rotate', #'rotate', #'zoom', 'rotate+zoom', or None        
        "augmentation_rate"     : 0.5, # 0.0 a 1.0 (Porcentagem do dataset que sofrerá augmentation)        
        "save_debug_images"     : False
    }, 
    
    # Controle de Outputs
    "save_weights": True,
    "weights_filename_loss": "incorMax2_loss.weights.keras",
    "weights_filename_auc": "incorMax2_auc.weights.keras"
}  

def get_experiment_group_folder(config):
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
        rate_str = f"{int(rate*100)}pct"
        active_flags.append(f"Aug_{aug}_{rate_str}")

    if not active_flags:
        return mode_prefix + "Baseline_Raw"
        
    return mode_prefix + "_".join(active_flags)

def save_execution_log(config, results, duration_str, timestamp):
    group_folder = get_experiment_group_folder(config)
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

def execute_single_training():
    """Executa um único treino isolado (chamado no subprocesso)."""
    from Classification3D.models.incor.incorDualRunner import run_incor_dual_training
    from Classification3D.models.incor.incorConcatRunner import run_incor_concat_training
    
    start_time = time.time()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    try:
        if EXPERIMENT_CONFIG.get("model_mode") == "early_channel" or EXPERIMENT_CONFIG.get("model_mode") == "concat_volume":
            training_results = run_incor_concat_training(EXPERIMENT_CONFIG, EXPERIMENT_CONFIG.get("model_mode"))
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
        
        print(f"Tempo Carregamento Dados: {str(datetime.timedelta(seconds=int(load_sec)))} ({load_sec:.2f}s)")
        print(f"Tempo de Treinamento Puro: {str(datetime.timedelta(seconds=int(train_sec)))} ({train_sec:.2f}s)")
        print(f"Tempo de Teste e Inferência: {str(datetime.timedelta(seconds=int(test_sec)))} ({test_sec:.2f}s)")
    
    save_execution_log(EXPERIMENT_CONFIG, training_results, duration_str, timestamp)

def run_orchestrator(total_runs):
    """Orquestra 10 execuções, cada uma em um processo Python totalmente novo."""
    print(f"=== Orquestrando Bateria de {total_runs} Processos Isolados: {EXPERIMENT_CONFIG['experiment_name']} ===")
    script_path = os.path.abspath(__file__)
    
    for run_idx in range(1, total_runs + 1):
        print(f"\n==========================================")
        print(f"   DISPARANDO SUBPROCESSO {run_idx}/{total_runs}")
        print(f"==========================================")
        
        # Dispara uma nova instância do Python do zero para garantir 100% de isolamento de memória e GPU
        cmd = [sys.executable, script_path, "--single-run"]
        result = subprocess.run(cmd)
        
        if result.returncode != 0:
            print(f"⚠️ AVISO: O subprocesso {run_idx} terminou com código de erro {result.returncode}")
            
        # Pausa leve entre processos
        time.sleep(2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Orquestrador de Experimentos 3D")
    parser.add_argument("--single-run", action="store_true", help="Executa apenas uma rodada de treino isolada")
    args = parser.parse_args()

    if args.single_run:
        execute_single_training()
    else:
        run_orchestrator(total_runs=9)