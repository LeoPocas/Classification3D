import tensorflow as tf

# Verifique se a GPU está disponível
physical_devices = tf.config.list_physical_devices('GPU')
print("GPUs Disponíveis: ", physical_devices)

if physical_devices:
    # Ative o crescimento de memória para as GPUs
    for gpu in physical_devices:
        tf.config.experimental.set_memory_growth(gpu, True)
    print("TensorFlow está utilizando a GPU.")
else:
    print("TensorFlow não detectou nenhuma GPU. Certifique-se de que os drivers da GPU e TensorFlow estão instalados corretamente.")
