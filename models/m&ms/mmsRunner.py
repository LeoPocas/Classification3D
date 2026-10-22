import gc
import os
import time

import numpy as np
from keras import mixed_precision
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from Classification3D.models.models import (
    dualInput_Resnet,
    singleInput_Resnet_ChannelConcat,
    singleInput_Resnet_Concat,
)
from Classification3D.preprocessing.load_mms import load_mms_data_dual_input
from Classification3D.utils import LABEL_MAPPING_MMS, MMs_REESPACADO, WEIGHT_PATH

mixed_precision.set_global_policy("float32")


def _load_data(training, preprocessing_config=None):
    data, labels = load_mms_data_dual_input(
        training=training,
        data_path=MMs_REESPACADO,
        label_mapping=LABEL_MAPPING_MMS,
        preprocessing_config=preprocessing_config,
    )
    return data, labels


def _join_inputs(data, model_mode):
    if model_mode == "concat_volume":
        return np.concatenate([data["systole"], data["diastole"]], axis=3)
    if model_mode == "early_channel":
        return np.concatenate([data["systole"], data["diastole"]], axis=-1)
    raise ValueError(
        f"Modo de modelo desconhecido: {model_mode}. "
        "Use 'dual', 'concat_volume' ou 'early_channel'."
    )


def _classification_report(y_true, y_pred):
    return classification_report(
        y_true,
        y_pred,
        labels=range(len(LABEL_MAPPING_MMS)),
        target_names=list(LABEL_MAPPING_MMS.keys()),
        output_dict=True,
        zero_division=0,
    )


def run_mms_training(config, model_mode):
    batch_size = config.get("batch_size", 8)
    epochs = config.get("epochs", 300)
    val_split_size = config.get("test_split", 0.2)
    random_seed = config.get("random_seed", 41)
    learning_rate = config.get("learning_rate", 0.0001)
    pred_batch_size = config.get("predictions_batch_size", 4)
    weights_loss_name = config.get("weights_filename_loss", "mms_loss.weights.keras")
    weights_auc_name = config.get("weights_filename_auc", "mms_auc.weights.keras")
    preprocessing_config = config.get("preprocessing", {})

    print("[RUNNER-MMS] Carregando dados M&Ms...")
    load_start = time.time()
    data, labels = _load_data(training=True, preprocessing_config=preprocessing_config)
    load_duration = time.time() - load_start
    print(f"[RUNNER-MMS] Dados carregados em {load_duration:.2f} segundos.")

    train_start = time.time()
    if model_mode == "dual":
        x_train_systole, x_val_systole, x_train_diastole, x_val_diastole, y_train, y_val = train_test_split(
            data["systole"],
            data["diastole"],
            labels,
            test_size=val_split_size,
            random_state=random_seed,
            stratify=np.argmax(labels, axis=1),
        )
        model = dualInput_Resnet(num_classes=len(LABEL_MAPPING_MMS))
        train_inputs = {
            "systole_input": x_train_systole,
            "diastole_input": x_train_diastole,
        }
        validation_inputs = {
            "systole_input": x_val_systole,
            "diastole_input": x_val_diastole,
        }
    else:
        x_combined = _join_inputs(data, model_mode)
        x_train, x_val, y_train, y_val = train_test_split(
            x_combined,
            labels,
            test_size=val_split_size,
            random_state=random_seed,
            stratify=np.argmax(labels, axis=1),
        )
        model = (
            singleInput_Resnet_Concat(
                input_shape=x_train.shape[1:], num_classes=len(LABEL_MAPPING_MMS)
            )
            if model_mode == "concat_volume"
            else singleInput_Resnet_ChannelConcat(
                input_shape=x_train.shape[1:], num_classes=len(LABEL_MAPPING_MMS)
            )
        )
        train_inputs = x_train
        validation_inputs = x_val

    del data
    gc.collect()

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy", "auc"],
    )
    callbacks = [
        ModelCheckpoint(
            os.path.join(WEIGHT_PATH, weights_loss_name),
            save_best_only=True,
            monitor="val_loss",
            mode="min",
        ),
        ModelCheckpoint(
            os.path.join(WEIGHT_PATH, weights_auc_name),
            save_best_only=True,
            monitor="val_auc",
            mode="max",
        ),
        ReduceLROnPlateau(monitor="val_loss", factor=0.97, patience=4, min_lr=1e-7),
        EarlyStopping(
            monitor="val_loss",
            mode="min",
            baseline=0.65,
            patience=50,
            start_from_epoch=100,
            verbose=1,
            restore_best_weights=True,
        )
    ]

    print(f"[RUNNER-MMS] Iniciando treinamento (Epochs: {epochs}, Batch: {batch_size})...")
    history = model.fit(
        train_inputs,
        y_train,
        validation_data=(validation_inputs, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=2,
    )
    train_duration = time.time() - train_start
    print(f"[RUNNER-MMS] Treinamento concluído em {train_duration:.2f} segundos.")

    del train_inputs, validation_inputs, y_train, y_val
    gc.collect()

    test_start = time.time()
    best_weights_path = os.path.join(WEIGHT_PATH, weights_loss_name)
    model.load_weights(best_weights_path)
    test_data, test_labels = _load_data(
        training=False, preprocessing_config=preprocessing_config
    )
    test_inputs = (
        {
            "systole_input": test_data["systole"],
            "diastole_input": test_data["diastole"],
        }
        if model_mode == "dual"
        else _join_inputs(test_data, model_mode)
    )
    del test_data
    gc.collect()

    results = model.evaluate(test_inputs, test_labels, batch_size=pred_batch_size, verbose=2)
    predictions = model.predict(test_inputs, batch_size=pred_batch_size)
    y_test_pred = np.argmax(predictions, axis=1)
    y_test_true = np.argmax(test_labels, axis=1)
    test_cm = confusion_matrix(
        y_test_true, y_test_pred, labels=range(len(LABEL_MAPPING_MMS))
    )
    test_cr = _classification_report(y_test_true, y_test_pred)
    test_duration = time.time() - test_start

    print("[RUNNER-MMS] Resultados no conjunto de teste:", results)
    print("\nMatriz de Confusão no conjunto de teste:\n", test_cm)
    print(f"[RUNNER-MMS] Fase de teste concluída em {test_duration:.2f} segundos.")

    del test_inputs, test_labels, predictions
    gc.collect()
    best_loss_index = int(np.argmin(history.history["val_loss"]))

    return {
        "timing": {
            "load_data_seconds": load_duration,
            "training_seconds": train_duration,
            "testing_seconds": test_duration,
        },
        "best_epoch": best_loss_index + 1,
        "best_val_loss": float(history.history["val_loss"][best_loss_index]),
        "best_val_accuracy": float(history.history["val_accuracy"][best_loss_index]),
        "best_val_auc": float(history.history["val_auc"][best_loss_index]),
        "last_epoch_executed": len(history.history["loss"]),
        "final_train_loss": float(history.history["loss"][-1]),
        "final_train_accuracy": float(history.history["accuracy"][-1]),
        "final_val_loss": float(history.history["val_loss"][-1]),
        "final_val_accuracy": float(history.history["val_accuracy"][-1]),
        "final_val_auc": float(history.history["val_auc"][-1]),
        "test_results_raw": results,
        "test_accuracy": results[1] if len(results) > 1 else None,
        "test_auc": results[2] if len(results) > 2 else None,
        "confusion_matrix": test_cm.tolist(),
        "classification_report": test_cr,
    }
