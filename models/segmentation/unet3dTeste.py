import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from unet3dModel import load_acdc_data_3d, unet_3d, weighted_dice_coefficient, weighted_categorical_crossentropy

# Número de classes
NUM_CLASSES = 4

# Carregar o modelo diretamente do arquivo e reconstruí-lo
model = unet_3d(input_size=(256, 256, 16, 1), num_classes=NUM_CLASSES)

# Carregar os pesos treinados
model.load_weights('./weights/unet.weights.keras')

# Compilar o modelo com a mesma configuração usada no treinamento
model.compile(optimizer='adam', loss='weighted_categorical_crossentropy', metrics=[weighted_dice_coefficient])

# Carregar os dados de teste
test_images, test_masks = load_acdc_data_3d('./ACDC/database/testing/')

# Fazer as predições no conjunto de teste
predictions = model.predict(test_images, verbose=1, batch_size=4)

# Transformar as predições para a forma de classes
predictions = np.argmax(predictions, axis=-1)

# Calcular F1-Score por classe
f1_scores = []
for class_id in range(NUM_CLASSES):
    y_true_class = (test_masks == class_id).astype(int)
    y_pred_class = (predictions == class_id).astype(int)
    f1 = f1_score(y_true_class.flatten(), y_pred_class.flatten())
    f1_scores.append(f1)

# Exibir F1-Score para cada classe
for i, f1 in enumerate(f1_scores):
    print(f'F1-Score para a classe {i}: {f1:.4f}')

# Calcular Accuracy geral
accuracy = accuracy_score(test_masks.flatten(), predictions.flatten())
print(f'Accuracy geral: {accuracy:.4f}')

# Matriz de confusão
conf_matrix = confusion_matrix(test_masks.flatten(), predictions.flatten(), labels=range(NUM_CLASSES))
print("\nMatriz de Confusão:")
print(conf_matrix)


for idx in range(len(test_images)//15):
    plt.figure(figsize=(12, 6))

    # Obter o índice do meio para a profundidade (eixo Z)
    mid_depth = test_images.shape[3] // 2  # Considera a profundidade da imagem 3D

    # Imagem original
    plt.subplot(1, 3, 1)
    plt.title("Imagem original")
    plt.imshow(test_images[idx, :, :, mid_depth, 0], cmap='gray')
    plt.axis('off')

    # Máscara verdadeira
    plt.subplot(1, 3, 2)
    plt.title("Máscara verdadeira")
    plt.imshow(test_masks[idx, :, :, mid_depth], cmap='jet')
    plt.axis('off')

    # Predição do modelo
    plt.subplot(1, 3, 3)
    plt.title("Predição")
    plt.imshow(predictions[idx, :, :, mid_depth], cmap='jet')
    plt.axis('off')

    plt.savefig(f'./output/prediction_class_image_{idx}.png')
    plt.close() 