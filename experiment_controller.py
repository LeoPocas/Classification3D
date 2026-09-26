import os
import sys
import time
import json
import datetime
import traceback
import subprocess
import argparse
from copy import deepcopy

TIMEOUT_SEGUNDOS = 7200  # 120 minutos
log_file_path = f"log_execucao_run_250926.txt"

# 1. FORÇA O KERAS/TENSORFLOW A ALOCAR VRAM DINAMICAMENTE (NÃO TUDO DE UMA VEZ)
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

# ==========================================
# CONFIGURAÇÃO DA EXECUÇÃO (CONTROLE CENTRAL)
# ==========================================
BASE_EXPERIMENT_CONFIG = {
    "model_mode": "concat", # 'dual', 'concat'/'concat_volume' ou 'early_channel'
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
        "apply_clahe"           : False,
        "normalization"         : None, #"min_max", #ou 'z_score', None
        "resampling"            : False,
        "augmentation"          : 'rotate', #'zoom', 'rotate+zoom', or None        
        "augmentation_rate"     : 1.0, # <= 1: probabilidade; > 1: número de cópias aumentadas        
        "save_debug_images"     : False
    }, 
    
    # Controle de Outputs
    "save_weights": True,
    "weights_filename_loss": "incorMax2_loss.weights.keras",
    "weights_filename_auc": "incorMax2_auc.weights.keras"
}

DEFAULT_PLANS_FILE = os.path.join(os.path.dirname(__file__), "experiment_plans.json")


def load_experiment_plans(plans_file):
    with open(plans_file, "r", encoding="utf-8") as file:
        plans = json.load(file)

    if not isinstance(plans, list):
        raise ValueError("O arquivo de planos deve conter uma lista JSON.")
    return plans


def merge_config(overrides):
    config = deepcopy(BASE_EXPERIMENT_CONFIG)
    config.update({key: value for key, value in overrides.items() if key != "preprocessing"})
    config["preprocessing"].update(overrides.get("preprocessing", {}))
    return config


def build_experiment_configs(experiment_plans, runs_override=None):
    configs = []
    for plan_number, plan in enumerate(experiment_plans, start=1):
        runs = runs_override if runs_override is not None else plan.get("runs", 1)
        if runs < 1:
            raise ValueError("O número de execuções deve ser maior ou igual a 1.")
        config = merge_config(plan.get("overrides", {}))
        configs.extend(
            (config, plan_number, run_number)
            for run_number in range(1, runs + 1)
        )
    return configs


def add_run_suffix(filename, plan_number, run_number):
    stem, extension = os.path.splitext(filename)
    return f"{stem}_plan_{plan_number}_run_{run_number}{extension}"

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
        rate = float(rate)
        rate_str = f"{rate:g}x" if rate > 1 else f"{int(rate*100)}pct"
        active_flags.append(f"Aug_{aug}_{rate_str}")

    if not active_flags:
        return mode_prefix + "seq_Baseline_Raw"
        
    return mode_prefix + "seq_" + "_".join(active_flags)

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

def check_gpu_availability():
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        raise RuntimeError(
            "CRÍTICO: Nenhuma GPU encontrada pelo TensorFlow! "
            "A execução foi abortada para evitar treino na CPU."
        )
    print(f"[CHECK] GPU detectada com sucesso: {gpus}")

def execute_single_training(config, run_number):
    """Executa um único treino isolado (chamado no subprocesso)."""

    check_gpu_availability()
    
    from Classification3D.models.incor.incorDualRunner import run_incor_dual_training
    from Classification3D.models.incor.incorConcatRunner import run_incor_concat_training
    
    start_time = time.time()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    try:
        model_mode = config.get("model_mode")
        if model_mode == "concat":
            model_mode = "concat_volume"

        if model_mode in {"early_channel", "concat_volume"}:
            training_results = run_incor_concat_training(config, model_mode)
        else:
            if model_mode != "dual":
                raise ValueError(
                    f"Modo de modelo desconhecido: {model_mode}. "
                    "Use 'dual', 'concat_volume' ou 'early_channel'."
                )
            training_results = run_incor_dual_training(config)
        
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
    
    save_execution_log(config, training_results, duration_str, timestamp)

def run_orchestrator(experiment_plans, runs_override=None):
    """Executa todos os planos, isolando cada rodada em um processo Python novo."""
    script_path = os.path.abspath(__file__)

    experiment_configs = build_experiment_configs(experiment_plans, runs_override)
    print(f"=== Orquestrando {len(experiment_configs)} Processos Isolados ===")

    for experiment_index, (config, plan_number, run_number) in enumerate(experiment_configs, start=1):
        run_config = deepcopy(config)
        run_config["weights_filename_loss"] = add_run_suffix(
            run_config["weights_filename_loss"], plan_number, run_number
        )
        run_config["weights_filename_auc"] = add_run_suffix(
            run_config["weights_filename_auc"], plan_number, run_number
        )

        print(f"\n==========================================")
        print(f"   DISPARANDO SUBPROCESSO {experiment_index}/{len(experiment_configs)}")
        print(f"   Experimento: {run_config['experiment_name']} | Rodada: {run_number}")
        print(f"==========================================")

        cmd = [
            sys.executable,
            "-u",
            script_path,
            "--single-run",
            "--run-number",
            str(run_number),
            "--experiment-json",
            json.dumps(run_config)
        ]

        try:
            with open(log_file_path, "a", encoding="utf-8") as f_log:
                f_log.write(f"\n--- INÍCIO EXP {experiment_index} (Plan {plan_number} Run {run_number}) [{datetime.datetime.now()}] ---\n")
                f_log.flush()

                result = subprocess.run(
                    cmd,
                    timeout=TIMEOUT_SEGUNDOS,
                    # Redirecionar a saída diretamente para ficheiro evita encher a memória pipe
                    stdout=f_log,               # Grava a saída diretamente no arquivo
                    stderr=subprocess.STDOUT    # Redireciona erros para o mesmo arquivo de log
                )

                if result.returncode != 0:
                    print(f"AVISO: O subprocesso {experiment_index} falhou (Código: {result.returncode})")

        except subprocess.TimeoutExpired:
            print(f"CRÍTICO: O treino {experiment_index} excedeu {TIMEOUT_SEGUNDOS}s e foi cancelado.")
            # Escreve o timeout no arquivo de log para registro
            with open(log_file_path, "a", encoding="utf-8") as f_log:
                f_log.write(f"\n[TIMEOUT] Treino cancelado após {TIMEOUT_SEGUNDOS} segundos.\n")
            
            # Força a limpeza de subprocessos remanescentes e avança para o próximo
            subprocess.run(["pkill", "-f", "single-run"])
            continue

        time.sleep(2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Orquestrador de Experimentos 3D")
    parser.add_argument("--single-run", action="store_true", help="Executa apenas uma rodada de treino isolada")
    parser.add_argument("--runs", type=int, help="Número de repetições para cada plano")
    parser.add_argument(
        "--plans",
        default=DEFAULT_PLANS_FILE,
        help="Arquivo JSON com a lista de planos de execução"
    )
    parser.add_argument("--run-number", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--experiment-json", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.single_run:
        if not args.experiment_json:
            raise ValueError("--experiment-json é obrigatório com --single-run.")
        execute_single_training(json.loads(args.experiment_json), args.run_number)
    else:
        run_orchestrator(load_experiment_plans(args.plans), runs_override=args.runs)