import pathlib
import glob
import tensorflow as tf
import pretty_midi
import numpy as np


def get_data():
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

    train_x = []
    j = 0
    # Preparing training data
    for file in filenames:
        if j == 10:
            break
        j += 1
        file_information = midi_to_notes(file)
        train_x.append(file_information)

    return train_x


# Extracting note info from sample midi file
def midi_to_notes(midi_file):
    pm = pretty_midi.PrettyMIDI(midi_file)
    instrument = pm.instruments[0]

    # Storing and returning information
    note_information = []

    i=0
    for _, note in enumerate(instrument.notes):
        if i == 64:
            break
        note_information.append([note.pitch, note.start, note.end - note.start, note.velocity])
        i += 1

    return note_information


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
