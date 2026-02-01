# Attempting multiple notes at once!
import pretty_midi
import tensorflow as tf
import numpy as np
import pathlib
import glob
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout
from tensorflow.keras.layers import BatchNormalization, Activation, LeakyReLU, UpSampling2D, Conv2D
from tensorflow.keras.models import Sequential, Model

seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

# Extract data file
data_dir = pathlib.Path('data/maestro-v2.0.0')
# data_dir = pathlib.Path('./data')

if not data_dir.exists():
  tf.keras.utils.get_file(
      'maestro-v2.0.0-midi.zip',
      origin='https://storage.googleapis.com/magentadata/datasets/maestro/v2.0.0/maestro-v2.0.0-midi.zip',
      extract=True,
      cache_dir='.', cache_subdir='data',
)

filenames = glob.glob(str(data_dir/'**/*.mid*'))
print('Number of files:', len(filenames))

# --------------------------------------------------------------------------------------------------------------
trainx = []

# Extracting note info from sample midi file
def midi_to_notes(midi_file):
  pm = pretty_midi.PrettyMIDI(midi_file)
  instrument = pm.instruments[0]

  # Storing and returning information
  note_information = []

  i=0
  for _, note in enumerate(instrument.notes):
    if i==64:
      break
    note_information.append([note.pitch, note.start, note.end - note.start, note.velocity])
    i+=1

  return note_information
  # if len(note_information) < 2000:
  #   return [False, note_information]
  # else:
  #   return [True, note_information]

# Generate midi file
def notes_to_midi(inst, out_file, instrument_name):
  notes = inst[:, 0]
  starts, durations = inst[:, 1], inst[:, 2]
  velocities = inst[:, 3]

  min_start = np.amin(starts)
  starts = starts - min_start

  pm = pretty_midi.PrettyMIDI()
  instrument = pretty_midi.Instrument(program=pretty_midi.instrument_name_to_program(instrument_name))

  for i in range(len(notes)):
    # Place note into instrument
    note = pretty_midi.Note(velocity=int(velocities[i]), pitch=int(notes[i]), start=starts[i], end=starts[i] + durations[i])
    instrument.notes.append(note)

  # print(instrument.notes)
  pm.instruments.append(instrument)
  pm.write(out_file)

# Model creation...
np.random.seed(3)
tf.random.set_seed(3)

#  generator
# Add dense layers to get to the dimensions that you want
generator = Sequential()
generator.add(Dense(128*4*4, input_dim=100, activation=LeakyReLU(0.2)))
generator.add(BatchNormalization())
generator.add(Reshape((4, 4, 128)))
generator.add(UpSampling2D())
generator.add(Conv2D(64, kernel_size=(2, 4), padding='same'))
generator.add(BatchNormalization())
generator.add(Activation(LeakyReLU(0.2)))
generator.add(UpSampling2D())
generator.add(Conv2D(1, kernel_size=(4,4), padding='same', activation='tanh'))

#  discriminator
discriminator = Sequential()
discriminator.add(Conv2D(64, kernel_size=(4,4), strides=4, input_shape=(16,16,1), padding="same"))
discriminator.add(Activation(LeakyReLU(0.2)))
discriminator.add(Dropout(0.3))
discriminator.add(Conv2D(128, kernel_size=(2,4), strides=4, padding="same"))
discriminator.add(Activation(LeakyReLU(0.2)))
discriminator.add(Dropout(0.3))
discriminator.add(Flatten())
discriminator.add(Dense(1, activation='sigmoid'))
discriminator.compile(loss='binary_crossentropy', optimizer='adam')
discriminator.trainable = False

#  GAN model
generator_input = Input(shape=(100,))
discriminator_output = discriminator(generator(generator_input))
gan = Model(generator_input, discriminator_output)
gan.compile(loss='binary_crossentropy', optimizer='adam')
gan.summary()

#  train
def gan_train(epoch, batch_size, saving_interval):

  trainx = []
  j=0
  # Preparing training data
  for file in filenames:
    if j==10:
      break
    j+=1
    file_information = midi_to_notes(file)
    trainx.append(file_information)

  # Changing the train data, sort by start time for each sample
  trainx = np.array(trainx)
  print(trainx.shape)
  for i in range(trainx.shape[0]):
    trainx[i] = sorted(trainx[i], key=lambda s: s[:][1])
  print(trainx.shape)
  # Normalize between -1 and 1
  max_note = np.amax(trainx[:][:][0])
  max_start = np.amax(trainx[:][:][1])
  max_duration = np.amax(trainx[:][:][2])
  max_velocity = np.amax(trainx[:][:][3])
  # print(max_note, max_start, max_duration, max_velocity)

  trainx[:][:][0] = (trainx[:][:][0] - (max_note / 2.)) / (max_note / 2.)
  trainx[:][:][1] = (trainx[:][:][1] - (max_start / 2.)) / (max_start / 2.)
  trainx[:][:][2] = (trainx[:][:][2] - (max_duration / 2.)) / (max_duration / 2.)
  trainx[:][:][3] = (trainx[:][:][3] - (max_velocity / 2.)) / (max_velocity / 2.)

  # Padding trainx so input size is the same and capped at 400 notes
  max_len = 64
  trainx_padded = []
  for i in range(len(trainx)):
    if len(trainx[i]) > max_len:
      trainx[i] = trainx[i][0:64][:]
    else:
      padCount = max_len - len(trainx[i])
      # trainx[i] = np.pad(trainx[i], pad_width=[(0, padCount), (0, 0)], mode='constant', constant_values=0)
      d = []
      for newdata in np.pad(trainx[i], pad_width=[(0, padCount), (0, 0)], mode='constant', constant_values=-1):
        d.append(newdata)

      trainx_padded.append(d)

  trainx_padded = np.array(trainx_padded)
  trainx_padded = trainx_padded.reshape(trainx_padded.shape[0], 16, 16, 1)
  print("Train data shape: " + str(trainx_padded.shape))

  true = np.ones((batch_size, 1))
  fake = np.zeros((batch_size, 1))

  for i in range(epoch):
      # train discriminator with real "images"
      idx = np.random.randint(0, trainx_padded.shape[0], batch_size)
      images = trainx_padded[idx]
      discriminator_loss_real = discriminator.train_on_batch(images, true)

      # train discriminator with fake "images"
      noise = np.random.normal(0, 1, (batch_size, 100))
      gen_images = generator.predict(noise, verbose=0)
      discriminator_loss_fake = discriminator.train_on_batch(gen_images, fake)

      # discriminator loss
      discriminator_loss = 0.5 * np.add(discriminator_loss_real, discriminator_loss_fake)

      # train gan with fake images - gan loss
      generator_loss = gan.train_on_batch(noise, true) # note that output is true

      if i % 1000 == 0:
          print('epoch:%d' % i, ' d_loss:%.4f' % discriminator_loss, ' g_loss:%.4f' % generator_loss)

      #  Saving images
      if i % saving_interval == 0:
        # r, c = 5, 5
        noise = np.random.normal(0, 1, (1, 100))
        gen_image = generator.predict(noise)

        gen_image = gen_image.reshape(64, 4)

        gen_image[:, 0] = (gen_image[:, 0] * 0.5 + 0.5) * max_note
        gen_image[:, 1] = (gen_image[:, 1] * 0.5 + 0.5) * max_start
        gen_image[:, 2] = (gen_image[:, 2] * 0.5 + 0.5) * max_duration
        gen_image[:, 3] = (gen_image[:, 3] * 0.5 + 0.5) * max_velocity

        fname = ('./OutputMaestro/%i.midi' % i)
        example_pm = notes_to_midi(gen_image, fname, 'Electric Grand Piano')

        if i % (saving_interval * 10) == 0:
          generator.save('./OutputMaestro/%i_generator.keras' % i)

gan_train(1001, 32, 1000)
