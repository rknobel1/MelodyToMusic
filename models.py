from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout
from tensorflow.keras.layers import BatchNormalization, Activation, LeakyReLU, UpSampling2D, Conv2D
from tensorflow.keras.models import Sequential, Model


class Generator(Model):
    def __init__(self, noise_dim=100):
        super().__init__()
        self.noise_dim = noise_dim

        self.dense = Dense(128 * 4 * 4)
        self.bn1 = BatchNormalization()
        self.reshape = Reshape((4, 4, 128))

        self.up1 = UpSampling2D()
        self.conv1 = Conv2D(64, kernel_size=(2, 4), padding="same")
        self.bn2 = BatchNormalization()

        self.up2 = UpSampling2D()
        self.conv2 = Conv2D(1, kernel_size=(4, 4),
                                   padding="same",
                                   activation="tanh")

        self.lrelu = LeakyReLU(0.2)

    def call(self, z, training=False):
        x = self.dense(z)
        x = self.lrelu(x)
        x = self.bn1(x, training=training)
        x = self.reshape(x)

        x = self.up1(x)
        x = self.conv1(x)
        x = self.lrelu(x)
        x = self.bn2(x, training=training)

        x = self.up2(x)
        return self.conv2(x)


class Discriminator(Model):
    def __init__(self):
        super().__init__()

        self.conv1 = Conv2D(64, kernel_size=(4, 4),
                                   strides=4,
                                   padding="same")
        self.drop1 = Dropout(0.3)

        self.conv2 = Conv2D(128, kernel_size=(2, 4),
                                   strides=4,
                                   padding="same")
        self.drop2 = Dropout(0.3)

        self.flatten = Flatten()
        self.out = Dense(1, activation="sigmoid")

        self.lrelu = LeakyReLU(0.2)

    def call(self, x, training=False):
        x = self.conv1(x)
        x = self.lrelu(x)
        x = self.drop1(x, training=training)

        x = self.conv2(x)
        x = self.lrelu(x)
        x = self.drop2(x, training=training)

        x = self.flatten(x)
        return self.out(x)


def GAN():
    noise_dim = 100

    generator = Generator(noise_dim)
    discriminator = Discriminator()

    discriminator.compile(
        optimizer="adam",
        loss="binary_crossentropy"
    )

    discriminator.trainable = False

    gan_input = Input(shape=(noise_dim,))
    fake_img = generator(gan_input)
    gan_output = discriminator(fake_img)

    gan = Model(gan_input, gan_output)
    gan.compile(
        optimizer="adam",
        loss="binary_crossentropy"
    )

    gan.summary()

    return [gan, generator, discriminator]
