import numpy as np
import os
import gc
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, Callback
from keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report
from keras import mixed_precision
from Classification3D.models.classification.models import build_med3d
from Classification3D.utils import LABEL_MAPPING_MMS, WEIGHT_PATH, OUTPUT_PATH, INCOR_RESAMPLED_PATH, TARGET_SHAPE, ZOOM, SPACING
from Classification3D.preprocessing.roiExtraction import get_ROI_distance_transform
from Classification3D.preprocessing.equalizacao import pad_or_crop_volume, apply_clahe

mixed_precision.set_global_policy('float32')

class SimpleConfusionMatrixCallback(Callback):
    def __init__(self, validation_data_with_filenames, batch_size, label_mapping_keys):
        super().__init__()
        self.x_val_img, self.y_val, self.val_filenames = validation_data_with_filenames
        self.batch_size = batch_size
        self.label_mapping_keys = label_mapping_keys

    def on_epoch_end(self, epoch, logs=None):
        if self.x_val_img is None or len(self.x_val_img) == 0:
            print(f"Callback: Sem dados de validação para a época {epoch + 1}.")
            return

        y_pred_probs = self.model.predict(self.x_val_img, batch_size=self.batch_size)
        y_pred_classes = np.argmax(y_pred_probs, axis=1)
        y_true_classes = np.argmax(self.y_val, axis=1)
        
        cm = confusion_matrix(y_true_classes, y_pred_classes)
       
        cr = classification_report(y_true_classes, y_pred_classes, target_names=self.label_mapping_keys, zero_division=0)

        misclassified_count = 0
        for i in range(len(y_true_classes)):
            if y_pred_classes[i] != y_true_classes[i]:
                misclassified_count +=1
                filename = self.val_filenames[i]
                true_label_name = self.label_mapping_keys[y_true_classes[i]]
                predicted_label_name = self.label_mapping_keys[y_pred_classes[i]]
                print(f"  Arquivo: {filename}, Verdadeiro: {true_label_name}, Previsto: {predicted_label_name}")
        if misclassified_count == 0:
            print("  Nenhum arquivo de validação classificado incorretamente.")
        print("--- Fim da Análise de Validação da Época ---")

print("Carregando dados de treino e validação (entrada única)...")
images, labels, filenames = load_incor_data_with_filenames(
    training=True,
    use_ed_phase=True,
    use_es_phase=True
)

if len(images) == 0:
    print("Nenhum dado de treino carregado. Verifique a função de carregamento e os caminhos.")
    exit()

x_train_img, x_val_img, y_train, y_val, filenames_train, filenames_val = train_test_split(
    images, labels, filenames, test_size=0.2, random_state=37
)
print(f"Dados de treino: {len(x_train_img)} amostras.")
print(f"Dados de validação: {len(x_val_img)} amostras.")

del images, labels, filenames
gc.collect()

model = build_med3d()
print("Modelo build_med3d construído.")

optimizer = Adam(learning_rate=0.0001) # Ajuste a taxa de aprendizado conforme necessário
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

class_names_sorted = [name for name, index in sorted(LABEL_MAPPING_MMS.items(), key=lambda item: item[1])]

callbacks_list = [
    ModelCheckpoint(
        os.path.join(WEIGHT_PATH, "single_input_best_val_accuracy.weights.keras"),
        save_best_only=True, monitor="val_accuracy", mode="max", verbose=1
    ),
    ModelCheckpoint(
        os.path.join(WEIGHT_PATH, "single_input_best_val_loss.weights.keras"),
        save_best_only=True, monitor="val_loss", mode="min", verbose=1
    ),
    ReduceLROnPlateau(monitor='val_loss', factor=0.95, patience=5, min_lr=1e-7, verbose=1), # Ajustado
    EarlyStopping(monitor='val_accuracy', mode='max', baseline=0.90, patience=50, verbose=1, restore_best_weights=True), # Ajustado
    SimpleConfusionMatrixCallback( # Descomente se quiser análise por época na validação
    validation_data_with_filenames=(x_val_img, y_val, filenames_val),
    batch_size=4, # Ajuste o batch_size para predição no callback
    label_mapping_keys=class_names_sorted
    )
]

# --- Treinamento do Modelo ---
print("Iniciando treinamento do modelo de entrada única...")
history = model.fit(
    x_train_img,
    y_train,
    validation_data=(x_val_img, y_val),
    epochs=100,
    batch_size=8,
    callbacks=callbacks_list,
    verbose=2
)

print("Treinamento concluído.")