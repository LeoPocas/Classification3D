from Classification3D.models.classification.simCLR import create_encoder, create_projection_head, SimCLR
from keras.optimizers import Adam

# Instancia o encoder e o modelo SimCLR
encoder = create_encoder()
projection_head = create_projection_head()
model = SimCLR(encoder, projection_head)

# Compila o modelo
optimizer = Adam(learning_rate=1e-3)
model.compile(optimizer)

# Depois de treinar, salvamos os pesos do encoder
encoder.save_weights("encoder_ssl.h5")