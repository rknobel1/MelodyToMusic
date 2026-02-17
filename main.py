# Attempting multiple notes at once!
from train import *

# Setting seed
# seed = 42
# tf.random.set_seed(seed)
# np.random.seed(seed)

# Setting parameters
epochs = 100
batch_size = 64
saving_interval = epochs / 5

num_files = 10      # To check if code will run or to run on reduced data

# Build GAN architecture
gan = build_gan(lr_G=1e-4, lr_D=1e-4, device="cuda")

# Train model
gan_train(gan, epochs=epochs, batch_size=batch_size, saving_interval=saving_interval,
          # num_files=num_files
          )
