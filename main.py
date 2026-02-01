# Attempting multiple notes at once!
from train import *

# Setting seed
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

# Setting parameters
epochs = 10_000
batch_size = 32
saving_interval = 1000
gan = build_gan(device="cuda")

# Train model
gan_train(gan, epochs=epochs, batch_size=batch_size, saving_interval=saving_interval)
