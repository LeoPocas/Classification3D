import numpy as np
import os
import gc
import time
from Classification3D.models.models import cnn_3d_model, build_med3d, newModel, dualInput_Resnet, build_med3d_with_ssl
from Classification3D.preprocessing.loadIncorParametrized import load_incor_dual_parametrized
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, Callback
from keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report
from keras import mixed_precision
from sklearn.preprocessing import StandardScaler
from Classification3D.utils import LABEL_MAPPING, ACDC_REESPACADO_TESTING, WEIGHT_PATH

# Configuração para usar precisão mista
mixed_precision.set_global_policy('float32')

class ConfusionMatrixCallback(Callback):
    def __init__(self, validation_data, batch_size):
        super().__init__()
        self.validation_data = validation_data
        self.batch_size = batch_size

    def on_epoch_end(self, epoch, logs=None):
        # Dados de validação: múltiplos inputs e labels
        x_val_systole, x_val_diastole, y_val = self.validation_data
        
        # Geração de previsões usando todos os inputs
        y_pred = self.model.predict(
            {'systole_input': x_val_systole, 'diastole_input': x_val_diastole},
            batch_size=self.batch_size
        )
        
        # Obter as classes preditas
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Obter as classes verdadeiras
        y_true = np.argmax(y_val, axis=1)
        
        # Gerar a matriz de confusão
        cm = confusion_matrix(y_true, y_pred_classes)
        print(f"\nMatriz de Confusão após época {epoch + 1}:\n", cm)
        
        # Gerar o relatório de classificação
        cr = classification_report(y_true, y_pred_classes, target_names=list(LABEL_MAPPING.keys()))
        print(f"\nRelatório de Classificação após época {epoch + 1}:\n", cr)

def run_incor_dual_training(config):
    """
    Função encapsulada para rodar o treinamento recebendo parâmetros de configuração.
    """
    
    # Extração de parâmetros do config (com defaults se não existirem)
    batch_size = config.get("batch_size", 8)
    epochs = config.get("epochs", 300)
    test_split_size = config.get("test_split", 0.10)
    random_seed = config.get("random_seed", 41)
    learning_rate = config.get("learning_rate", 0.0001)
    pred_batch_size = config.get("predictions_batch_size", 4)
    weights_loss_name = config.get("weights_filename_loss", "incorMax2_loss.weights.keras")
    weights_auc_name = config.get("weights_filename_auc", "incorMax2_auc.weights.keras")
    
    # Parâmetros de pré-processamento vindos do config
    preprocess_params = config.get("preprocessing", {})
    
    print("[RUNNER] Carregando dados Incor Dual Parametrizados...")
    load_start = time.time()

    data, labels, _ = load_incor_dual_parametrized(training=True, preprocessing_config=preprocess_params)
    load_duration = time.time() - load_start
    print(f"[RUNNER] Dados carregados em {load_duration:.2f} segundos.")

    train_start = time.time()   
    
    systole_images = data['systole']
    diastole_images = data['diastole']

    x_train_systole, x_val_systole, x_train_diastole, x_val_diastole, y_train, y_val = train_test_split(
        systole_images, diastole_images, labels, test_size=test_split_size, random_state=random_seed
    )

    print("[RUNNER] Construindo Modelo...")
    # model = build_med3d() # Opção antiga comentada
    model = dualInput_Resnet()
    # model.summary() # Pode ser verboso demais para logs, descomentar se necessário

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy', 'auc'])

    callbacks = [
        ModelCheckpoint(os.path.join(WEIGHT_PATH, weights_loss_name), save_best_only=True, monitor="val_loss", mode="min"),
        ModelCheckpoint(os.path.join(WEIGHT_PATH, weights_auc_name), save_best_only=True, monitor="val_auc", mode="max"),
        ReduceLROnPlateau(monitor='val_loss', factor=0.97, patience=4, min_lr=1e-7),
        EarlyStopping(monitor='val_loss', mode='min', baseline=0.99, patience=600, verbose=1, restore_best_weights=True)
    ]

    print(f"[RUNNER] Iniciando Treinamento (Epochs: {epochs}, Batch: {batch_size})...")

    history = model.fit(
        {'systole_input': x_train_systole, 'diastole_input': x_train_diastole}, 
        y_train,
        validation_data=(
        {'systole_input': x_val_systole, 'diastole_input': x_val_diastole}, y_val), 
        epochs=epochs, batch_size=batch_size,
        callbacks=callbacks,
        verbose=2
    )
    train_duration = time.time() - train_start
    print(f"[RUNNER] Treinamento concluído em {train_duration:.2f} segundos.")
    
    # Limpeza de memória
    del x_train_systole, x_val_systole, x_train_diastole, x_val_diastole, y_train, y_val
    gc.collect()

    print("[RUNNER] Iniciando Fase de Teste...")
    test_start = time.time()
    
    # Carregamento de teste parametrizado
    test_data, test_labels, test_filenames = load_incor_dual_parametrized(training=False, preprocessing_config=preprocess_params)
    
    test_systole = test_data['systole']
    test_diastole = test_data['diastole']

    # Avaliação
    results = model.evaluate(
        {'systole_input': test_systole, 'diastole_input': test_diastole}, 
        test_labels,
        batch_size=pred_batch_size,
        verbose=1
    )

    print("[RUNNER] Resultados no conjunto de teste:", results)

    y_test_pred = model.predict(
        {'systole_input': test_systole, 'diastole_input': test_diastole}, 
        batch_size=pred_batch_size
    )
    y_test_pred_classes = np.argmax(y_test_pred, axis=1)
    y_test_true = np.argmax(test_labels, axis=1)

    # Métricas Finais
    test_cm = confusion_matrix(y_test_true, y_test_pred_classes)
    
    test_duration = time.time() - test_start
    print(f"[RUNNER] Fase de teste concluída em {test_duration:.2f} segundos.")

    print("\nMatriz de Confusão no conjunto de teste:\n", test_cm)

    test_cr = classification_report(y_test_true, y_test_pred_classes, target_names=list(LABEL_MAPPING.keys()), output_dict=True)
    # print("\nRelatório de Classificação no conjunto de teste:\n", test_cr) # output_dict para salvar no JSON

    gc.collect()
    
    # Retornar dicionário com resultados importantes para o log
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
    # Configuração Padrão para execução direta (Legacy Mode)
    default_config = {
        "epochs": 300, 
        "batch_size": 8,
        "test_split": 0.10,
        "predictions_batch_size": 4,
        "weights_filename_loss": "incorMax2_loss.weights.keras",
        "weights_filename_auc": "incorMax2_auc.weights.keras"
    }
    print("Executando incorDualTraining diretamente (modo legado)...")
    run_incor_dual_training(default_config)
