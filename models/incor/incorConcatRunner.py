import numpy as np
import os
import gc
import time
from Classification3D.models.models import singleInput_Resnet_Concat
from Classification3D.preprocessing.loadIncorParametrized import load_incor_dual_parametrized
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, Callback
from keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report
from keras import mixed_precision
from Classification3D.utils import LABEL_MAPPING, WEIGHT_PATH

# Configuração para usar precisão mista (Mantido igual ao original)
mixed_precision.set_global_policy('float32')

class ConfusionMatrixCallback(Callback):
    """
    Callback adaptado para receber um único input concatenado (x_val)
    em vez do formato dual por dicionário.
    """
    def __init__(self, validation_data, batch_size):
        super().__init__()
        self.validation_data = validation_data
        self.batch_size = batch_size

    def on_epoch_end(self, epoch, logs=None):
        # Dados de validação: input único concatenado e labels
        x_val, y_val = self.validation_data
        
        # Geração de previsões usando o input unificado
        y_pred = self.model.predict(
            x_val,
            batch_size=self.batch_size,
            verbose=0 # verbose=0 para não poluir o log a cada época
        )
        
        # Obter as classes preditas e verdadeiras
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true = np.argmax(y_val, axis=1)
        
        # Gerar a matriz de confusão
        cm = confusion_matrix(y_true, y_pred_classes)
        print(f"\nMatriz de Confusão após época {epoch + 1}:\n", cm)
        
        # Gerar o relatório de classificação
        cr = classification_report(y_true, y_pred_classes, target_names=list(LABEL_MAPPING.keys()))
        print(f"\nRelatório de Classificação após época {epoch + 1}:\n", cr)

def run_incor_concat_training(config):
    """
    Função encapsulada para rodar o treinamento com input concatenado,
    mantendo paridade total de logs e métricas com o modelo original.
    """
    # Extração de parâmetros do config (Mantendo os mesmos defaults do original)
    batch_size = config.get("batch_size", 8)
    epochs = config.get("epochs", 300)
    val_split_size = config.get("test_split", 0.10) # Tratado como split de validação
    random_seed = config.get("random_seed", 41)
    learning_rate = config.get("learning_rate", 0.0001)
    pred_batch_size = config.get("predictions_batch_size", 4)
    weights_loss_name = config.get("weights_filename_loss", "incorConcat_loss.weights.keras")
    weights_auc_name = config.get("weights_filename_auc", "incorConcat_auc.weights.keras")
    
    preprocess_params = config.get("preprocessing", {})
    
    print("[RUNNER-CONCAT] Carregando dados Incor para concatenação...")
    load_start = time.time()

    data, labels, _ = load_incor_dual_parametrized(training=True, preprocessing_config=preprocess_params)
    
    # Concatena Sístole e Diástole no eixo Z (profundidade/slices)
    # De (N, 192, 192, 12) + (N, 192, 192, 12) para (N, 192, 192, 24)
    x_combined = np.concatenate([data['systole'], data['diastole']], axis=3)
    
    load_duration = time.time() - load_start
    print(f"[RUNNER-CONCAT] Dados carregados e concatenados em {load_duration:.2f} segundos.")

    train_start = time.time()   
    
    # Split de treino/validação usando a sugestão de estratificação
    x_train, x_val, y_train, y_val = train_test_split(
        x_combined, labels, test_size=val_split_size, random_state=random_seed, stratify=labels
    )
    
    # Liberar memória do array combinado bruto
    del x_combined, data
    gc.collect()

    print("[RUNNER-CONCAT] Construindo Modelo de Input Único...")
    # Define o input_shape dinamicamente baseado nos dados reais (ex: 192, 192, 24)
    input_shape = x_train.shape[1:] 
    model = singleInput_Resnet_Concat(input_shape=input_shape)

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy', 'auc'])

    # Restaurado os dois checkpoints originais (Loss e AUC) + o callback customizado por época
    callbacks = [
        ModelCheckpoint(os.path.join(WEIGHT_PATH, weights_loss_name), save_best_only=True, monitor="val_loss", mode="min"),
        ModelCheckpoint(os.path.join(WEIGHT_PATH, weights_auc_name), save_best_only=True, monitor="val_auc", mode="max"),
        ReduceLROnPlateau(monitor='val_loss', factor=0.97, patience=4, min_lr=1e-7)
        # ConfusionMatrixCallback(validation_data=(x_val, y_val), batch_size=batch_size)
    ]

    print(f"[RUNNER-CONCAT] Iniciando Treinamento (Epochs: {epochs}, Batch: {batch_size})...")

    history = model.fit(
        x_train, 
        y_train,
        validation_data=(x_val, y_val), 
        epochs=epochs, 
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=2
    )
    
    train_duration = time.time() - train_start
    print(f"[RUNNER-CONCAT] Treinamento concluído in {train_duration:.2f} segundos.")
    
    # Limpeza rigorosa de memória de treino para evitar OOM antes do teste
    del x_train, x_val, y_train, y_val
    gc.collect()

    print("[RUNNER-CONCAT] Iniciando Fase de Teste...")
    test_start = time.time()
    
    # Carregamento e preparação do conjunto de teste independente
    test_data, test_labels, _ = load_incor_dual_parametrized(training=False, preprocessing_config=preprocess_params)
    x_test_concat = np.concatenate([test_data['systole'], test_data['diastole']], axis=3)
    
    del test_data
    gc.collect()

    # Avaliação final
    results = model.evaluate(
        x_test_concat, 
        test_labels,
        batch_size=pred_batch_size,
        verbose=1
    )

    print("[RUNNER-CONCAT] Resultados no conjunto de teste:", results)

    y_test_pred = model.predict(x_test_concat, batch_size=pred_batch_size)
    y_test_pred_classes = np.argmax(y_test_pred, axis=1)
    y_test_true = np.argmax(test_labels, axis=1)

    # Métricas Finais de Teste
    test_cm = confusion_matrix(y_test_true, y_test_pred_classes)
    test_cr = classification_report(y_test_true, y_test_pred_classes, target_names=list(LABEL_MAPPING.keys()), output_dict=True)
    
    test_duration = time.time() - test_start
    print(f"[RUNNER-CONCAT] Fase de teste concluída em {test_duration:.2f} segundos.")
    print("\nMatriz de Confusão no conjunto de teste:\n", test_cm)

    del x_test_concat, test_labels
    gc.collect()
    
    # Retorna o dicionário com a estrutura idêntica à do código original
    return {
        "timing": {
            "load_data_seconds": load_duration,
            "training_seconds": train_duration,
            "testing_seconds": test_duration
        },
        "final_train_loss": history.history['loss'][-1],
        "final_train_accuracy": history.history['accuracy'][-1],
        "final_val_loss": history.history['val_loss'][-1],
        "final_val_accuracy": history.history['val_accuracy'][-1],
        "test_results_raw": results, # [loss, accuracy, auc]
        "test_accuracy": results[1] if len(results) > 1 else None,
        "test_auc": results[2] if len(results) > 2 else None,
        "confusion_matrix": test_cm.tolist(),
        "classification_report": test_cr
    }

if __name__ == "__main__":
    # Configuração Padrão para execução direta (Igual ao original, adaptando nomes de arquivos)
    default_config = {
        "epochs": 300, 
        "batch_size": 8,
        "test_split": 0.10,
        "predictions_batch_size": 4,
        "weights_filename_loss": "incorConcat_loss.weights.keras",
        "weights_filename_auc": "incorConcat_auc.weights.keras"
    }
    print("Executando incorConcatTraining diretamente...")
    run_incor_concat_training(default_config)