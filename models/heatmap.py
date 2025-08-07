import cv2
import numpy as np
import matplotlib.pyplot as plt
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore

def generate_and_save_gradcam_3d(model, systole_volume, diastole_volume, pred_class_idx, class_names, output_filename):
    """
    Gera e salva heatmaps Grad-CAM, redimensionando o heatmap para corresponder
    ao tamanho da imagem original.
    """
    score = CategoricalScore([pred_class_idx])
    gradcam = Gradcam(model, model_modifier=ReplaceToLinear(), clone=False)
    input_tensors = [systole_volume, diastole_volume]
    layer_name = 'conv3d_21'

    try:
        cam = gradcam(score,
                      seed_input=input_tensors,
                      penultimate_layer=layer_name)
    except Exception as e:
        print(f"\n--- ERRO ao gerar Grad-CAM ---")
        print(f"Erro: {e}")
        return

    fig, axs = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle(f"Grad-CAM para a classe predita: '{class_names[pred_class_idx]}'", fontsize=20)
    
    volume_to_slice = systole_volume[0]
    heatmap_array = cam[0]

    for i in range(12):
        row, col = divmod(i, 4)
        ax = axs[row, col]

        original_slice_gray = volume_to_slice[:, :, i, 0]
        heatmap_slice = heatmap_array[0, i]

        target_size = (original_slice_gray.shape[1], original_slice_gray.shape[0]) # (largura, altura)
        heatmap_slice_resized = cv2.resize(heatmap_slice, target_size, interpolation=cv2.INTER_LINEAR)
        heatmap_slice_normalized = (heatmap_slice_resized - np.min(heatmap_slice_resized)) / (np.max(heatmap_slice_resized) - np.min(heatmap_slice_resized) + 1e-8)
        heatmap_colored = np.uint8(plt.cm.jet(heatmap_slice_normalized)[..., :3] * 255)
        original_slice_rgb = cv2.cvtColor(original_slice_gray.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        superimposed_img = cv2.addWeighted(heatmap_colored, 0.4, original_slice_rgb, 0.6, 0)
        
        ax.imshow(superimposed_img)
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(f"Slice {i+1}")
        ax.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_filename, dpi=200)
    print(f"\nGrad-CAM salvo em: {output_filename}")
    plt.close(fig)