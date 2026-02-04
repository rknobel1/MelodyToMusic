# Attempting multiple notes at once!
from train import *

# Setting seed
# seed = 42
# tf.random.set_seed(seed)
# np.random.seed(seed)

# Setting parameters
epochs = 1_000_000
batch_size = 32
saving_interval = epochs / 5
num_files = 2_000

# Build GAN architecture
gan = build_gan(lr_G=1e-5, lr_D=1e-5, device="cuda")

# Train model
gan_train(gan, epochs=epochs, batch_size=batch_size, saving_interval=saving_interval, num_files=num_files)
