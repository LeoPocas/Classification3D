import cv2
import numpy as np
import matplotlib.pyplot as plt
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore
import math

def generate_and_save_gradcam_3d(model, systole_volume, diastole_volume, pred_class_idx, class_names, output_filename, layer_name='re_lu_3'):
    """
    Gera e salva heatmaps Grad-CAM. Versão final com a indexação de eixos
    corrigida para o output do tf-keras-vis (batch, H, W, D).
    """
    score = CategoricalScore([pred_class_idx])
    gradcam = Gradcam(model, model_modifier=ReplaceToLinear(), clone=False)
    input_tensors = [systole_volume, diastole_volume]

    try:
        cam = gradcam(score,
                      seed_input=input_tensors,
                      penultimate_layer=layer_name)
    except Exception as e:
        print(f"\n--- ERRO ao gerar Grad-CAM para a camada '{layer_name}' ---")
        print(f"Erro: {e}")
        return

    heatmap_array = cam[0] # Saída do heatmap para o primeiro input
    
    # --- CORREÇÃO DE INDEXAÇÃO ---
    # O shape do heatmap é (batch, altura, largura, profundidade).
    # Pegamos o número de slices da ÚLTIMA dimensão espacial (eixo 3).
    num_slices = heatmap_array.shape[3]
    
    cols = 4
    rows = math.ceil(num_slices / cols)
    
    print(f"Camada '{layer_name}' selecionada. Gerando heatmap com {num_slices} slices em uma grade de {rows}x{cols}.")

    fig, axs = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axs = axs.flatten()
    
    fig.suptitle(f"Grad-CAM para '{class_names[pred_class_idx]}' (Camada: {layer_name})", fontsize=20)
    
    volume_to_slice = systole_volume[0]

    for i in range(num_slices):
        ax = axs[i]

        original_slice_gray = volume_to_slice[:, :, i, 0]
        
        # --- CORREÇÃO DE INDEXAÇÃO ---
        # Extrai o slice 'i' da dimensão de profundidade (o último eixo)
        heatmap_slice = heatmap_array[0, :, :, i]
        
        target_size = (original_slice_gray.shape[1], original_slice_gray.shape[0])
        heatmap_slice_resized = cv2.resize(heatmap_slice, target_size, interpolation=cv2.INTER_LINEAR)
        
        heatmap_slice_normalized = (heatmap_slice_resized - np.min(heatmap_slice_resized)) / (np.max(heatmap_slice_resized) - np.min(heatmap_slice_resized) + 1e-8)
        heatmap_colored = np.uint8(plt.cm.jet(heatmap_slice_normalized)[..., :3] * 255)
        original_slice_rgb = cv2.cvtColor(original_slice_gray.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        superimposed_img = cv2.addWeighted(heatmap_colored, 0.5, original_slice_rgb, 0.5, 0)
        
        ax.imshow(superimposed_img)
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(f"Slice {i+1} (Heatmap)")
        ax.axis('off')
    
    for j in range(num_slices, len(axs)):
        axs[j].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_filename, dpi=200)
    print(f"\nGrad-CAM salvo em: {output_filename}")
    plt.close(fig)