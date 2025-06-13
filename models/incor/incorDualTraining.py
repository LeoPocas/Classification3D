import numpy as np
import os
import gc
from Classification3D.models.models import cnn_3d_model, build_med3d, newModel, dualInput_Resnet, build_med3d_with_ssl
from Classification3D.preprocessing.loadIncor import load_incor_data, load_incor_dual
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, Callback
from keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report
from keras import mixed_precision
from sklearn.preprocessing import StandardScaler
from Classification3D.utils import LABEL_MAPPING, ACDC_REESPACADO_TESTING, WEIGHT_PATH

# Configuração para usar precisão mista
mixed_precision.set_global_policy('float32')

# Configuração da GPU (opcional)
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#             tf.config.experimental.set_virtual_device_configuration(
#                 gpu,
#                 [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=15500)])
#     except RuntimeError as e:
#         print(e)

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

# images, labels = load_incor_data()
data, labels = load_incor_dual() 
systole_images = data['systole']
diastole_images = data['diastole']

# x_train_img, x_val_img, y_train, y_val = train_test_split(
#     images, labels, test_size=0.2, random_state=37
# )
# x_train_systole, x_val_systole, x_train_diastole, x_val_diastole, y_train, y_val= train_test_split(
#     systole_images, diastole_images, labels, test_size=0.2, random_state=37
# ) Bom

x_train_systole, x_val_systole, x_train_diastole, x_val_diastole, y_train, y_val= train_test_split(
    systole_images, diastole_images, labels, test_size=0.1, random_state=41
)

# model = build_med3d()
model = dualInput_Resnet()

# Compilar o modelo
optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# callbacks = [
#     ModelCheckpoint(WEIGHT_PATH + "incor_resnet.weights.keras", save_best_only=True, monitor="val_accuracy", mode="max"),
#     ReduceLROnPlateau(monitor='val_loss', factor=0.99, patience=4, min_lr=5e-7),
#     ConfusionMatrixCallback(validation_data=(x_val_img, y_val), batch_size=5),
#     EarlyStopping(monitor='val_accuracy', mode='max', baseline=0.96, patience=600, verbose=1, restore_best_weights=True)  
# ]
callbacks = [
    ModelCheckpoint(WEIGHT_PATH + "incor2_loss.weights.keras", save_best_only=True, monitor="val_loss", mode="min"),
    ModelCheckpoint(WEIGHT_PATH + "incor2_accuracy.weights.keras", save_best_only=True, monitor="val_accuracy", mode="max"),
    ReduceLROnPlateau(monitor='val_loss', factor=0.98, patience=3, min_lr=1e-7),
    EarlyStopping(monitor='val_loss', mode='min', baseline=0.99, patience=600, verbose=1, restore_best_weights=True)
    # ConfusionMatrixCallback(
    #     validation_data=(x_val_systole, x_val_diastole, y_val),  # Adicionar todos os inputs e labels
    #     batch_size=2
    # )
]

# history = model.fit(
#     x_train_img,
#     y_train,
#     validation_data=(x_val_img, y_val),
#     epochs=500, batch_size=5,
#     callbacks=callbacks,
#     verbose=2
# )

history = model.fit(
    {'systole_input': x_train_systole, 'diastole_input': x_train_diastole}, 
    y_train,
    validation_data=(
    {'systole_input': x_val_systole, 'diastole_input': x_val_diastole}, y_val), 
    epochs=300, batch_size=10,
    callbacks=callbacks,
    verbose=2
)
del x_train_systole, x_val_systole, x_train_diastole, x_val_diastole, y_train, y_val
gc.collect()

# Testar o modelo com os dados de teste
# test_images, test_labels = load_incor_data(training=False)
test_data, test_labels = load_incor_dual(training=False) 
test_systole = test_data['systole']
test_diastole = test_data['diastole']

# results_train = model.evaluate(x_val_img, y_val, verbose=0)
results = model.evaluate(
    {'systole_input': test_systole, 'diastole_input': test_diastole}, 
    test_labels,
    verbose=1
)

# print("Resultados do melhor modelo no conjunto de treino:", results_train)

# del x_train_img, x_val_img, y_train, y_val
# gc.collect()

# results = model.evaluate(test_images, test_labels, verbose=1)

print("Resultados no conjunto de teste:", results)

y_test_pred = model.predict({'systole_input': test_systole, 'diastole_input': test_diastole}, batch_size=4)
y_test_pred_classes = np.argmax(y_test_pred, axis=1)
y_test_true = np.argmax(test_labels, axis=1)

# Gerar matriz de confusão
test_cm = confusion_matrix(y_test_true, y_test_pred_classes)
print("\nMatriz de Confusão no conjunto de teste:\n", test_cm)

# Relatório de classificação
test_cr = classification_report(y_test_true, y_test_pred_classes, target_names=list(LABEL_MAPPING.keys()))
print("\nRelatório de Classificação no conjunto de teste:\n", test_cr)

gc.collect()