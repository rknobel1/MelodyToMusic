# Attempting multiple notes at once!
from train import *

# Setting seed
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

# Setting parameters
epochs = 100_000
batch_size = 32
saving_interval = epochs / 20
num_files = 5_000

# Build GAN architecture
gan = build_gan(device="cuda")

# Train model
gan_train(gan, epochs=epochs, batch_size=batch_size, saving_interval=saving_interval, num_files=num_files)
