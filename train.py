from models import *
from utils import *


def gan_train(gan, generator, discriminator, epoch=1_000_000, batch_size=32, saving_interval=1_000):
    print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

    # Extracting data
    train_x = get_data()

    # Changing the train data, sort by start time for each sample
    train_x = np.array(train_x)
    for i in range(train_x.shape[0]):
        train_x[i] = sorted(train_x[i], key=lambda s: s[:][1])
    print("Data shape: ", train_x.shape)

    # Normalize between -1 and 1
    max_note = np.amax(train_x[:][:][0])
    max_start = np.amax(train_x[:][:][1])
    max_duration = np.amax(train_x[:][:][2])
    max_velocity = np.amax(train_x[:][:][3])

    train_x[:][:][0] = (train_x[:][:][0] - (max_note / 2.)) / (max_note / 2.)
    train_x[:][:][1] = (train_x[:][:][1] - (max_start / 2.)) / (max_start / 2.)
    train_x[:][:][2] = (train_x[:][:][2] - (max_duration / 2.)) / (max_duration / 2.)
    train_x[:][:][3] = (train_x[:][:][3] - (max_velocity / 2.)) / (max_velocity / 2.)

    # Padding train_x so input size is the same and capped at 400 notes
    max_len = 64
    train_x_padded = []
    for i in range(len(train_x)):
        if len(train_x[i]) > max_len:
            train_x[i] = train_x[i][0:64][:]
        else:
            pad_count = max_len - len(train_x[i])
            d = []
            for new_data in np.pad(train_x[i], pad_width=[(0, pad_count), (0, 0)], mode='constant', constant_values=-1):
                d.append(new_data)

            train_x_padded.append(d)

    train_x_padded = np.array(train_x_padded)
    train_x_padded = train_x_padded.reshape(train_x_padded.shape[0], 16, 16, 1)
    print("Train data shape: " + str(train_x_padded.shape))

    true = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    for i in range(epoch + 1):
        # train discriminator with real music
        idx = np.random.randint(0, train_x_padded.shape[0], batch_size)
        images = train_x_padded[idx]
        discriminator_loss_real = discriminator.train_on_batch(images, true)

        # train discriminator with fake music
        noise = np.random.normal(0, 1, (batch_size, 100))
        gen_images = generator.predict(noise, verbose=0)
        discriminator_loss_fake = discriminator.train_on_batch(gen_images, fake)

        # discriminator loss
        discriminator_loss = 0.5 * np.add(discriminator_loss_real, discriminator_loss_fake)

        # train gan with fake music - gan loss
        generator_loss = gan.train_on_batch(noise, true) # note that output is true

        if i % 1000 == 0:
            print('epoch:%d' % i, ' d_loss:%.4f' % discriminator_loss, ' g_loss:%.4f' % generator_loss)

        #  Saving images
        if i % saving_interval == 0:
            noise = np.random.normal(0, 1, (1, 100))
            gen_image = generator.predict(noise)

            gen_image = gen_image.reshape(64, 4)

            gen_image[:, 0] = (gen_image[:, 0] * 0.5 + 0.5) * max_note
            gen_image[:, 1] = (gen_image[:, 1] * 0.5 + 0.5) * max_start
            gen_image[:, 2] = (gen_image[:, 2] * 0.5 + 0.5) * max_duration
            gen_image[:, 3] = (gen_image[:, 3] * 0.5 + 0.5) * max_velocity

            file_name = ('./OutputMaestro/%i.midi' % i)
            notes_to_midi(gen_image, file_name, 'Electric Grand Piano')

            if i % (saving_interval * 10) == 0:
                generator.save('./OutputMaestro/%i_generator.keras' % i)
