import os
import json
import glob
import pathlib
import pandas as pd
import numpy as np

import music21 as m21

import tensorflow as tf
from keras.layers import Dense, LSTM, Dropout
from keras import Input, Model
from keras.optimizers import Adam
from keras.losses import SparseCategoricalCrossentropy
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import to_categorical


# DATASET_PATH = '/content/data/maestro-v2.0.0/2018'
# DATASET_PATH = '/content/drive/MyDrive/deutschl/erk'
DATA_PATH = "file_dataset.txt"
MAPPING_PATH = "mapping.json"
MODEL_PATH = "model_rl.h5"

SEQUENCE_LENGTH = 64
NUM_PITCHES = 130
BATCH_SIZE = 32

ACCEPTABLE_DURATIONS = [0.25,  0.5,  0.75, 1.0, 1.5, 2, 3, 4]

# Data preparation
def convert_songs_to_int(songs, mapping_path=MAPPING_PATH):
    int_songs = []

    with open(MAPPING_PATH, "r") as fp:
      mappings = json.load(fp)

    songs = songs.split()
    for symbol in songs:
      int_songs.append(mappings[symbol])

    return int_songs

def generate_sequences_generator(sequence_length, batch_size, file_path=DATA_PATH):
    with open(file_path, "r") as fp:
        songs = fp.read()

    int_songs = convert_songs_to_int(songs)
    num_sequences = len(int_songs) - sequence_length
    num_batches = num_sequences // batch_size

    while True:
        for batch_idx in range(num_batches):
            batch_inputs = []
            batch_targets = []
            for i in range(batch_size):
                start_idx = batch_idx * batch_size + i
                inputs = int_songs[start_idx:start_idx + sequence_length]
                target = int_songs[start_idx + sequence_length]
                inputs = to_categorical(inputs, num_classes=NUM_PITCHES)
                batch_inputs.append(inputs)
                batch_targets.append(target)
            yield np.array(batch_inputs), np.array(batch_targets)

# Model
def NoteRNN(output_units, num_units=100, lr=0.001):
    input = Input(shape=(None, output_units))

    x = LSTM(num_units)(input)
    x = Dropout(0.2)(x)

    output = Dense(output_units, activation="softmax")(x)

    model = Model(input, output)

    model.compile(loss=SparseCategoricalCrossentropy(),
                  optimizer=Adam(learning_rate=lr),
                  metrics=["accuracy"])

    return model

model = NoteRNN(NUM_PITCHES)
model.load_weights(MODEL_PATH)

# Generate melody
with open(DATA_PATH, "r") as fp:
  songs = fp.read()

with open(MAPPING_PATH, "r") as fp:
  mappings = json.load(fp)

int_songs = convert_songs_to_int(songs)

i = 426602
# i = np.random.randint(0, len(int_songs) - SEQUENCE_LENGTH)
seed = songs[i-1:i + 99]


def _sample_with_temperature(probabilites, temperature=2):
  predictions = np.log(probabilites) / temperature
  probabilites = np.exp(predictions) / np.sum(np.exp(predictions))

  choices = range(len(probabilites))
  index = np.random.choice(choices, p=probabilites)

  return index

def generate_melody(input_sequence, num_steps, length=SEQUENCE_LENGTH, temperature=2):
  input_sequence = input_sequence.split()
  melody = input_sequence

  input_sequence = ["r"] * length + input_sequence

  input_sequence = [mappings[symbol] for symbol in input_sequence]

  num_off_notes = 0

  for _ in range(num_steps):
    input_sequence = input_sequence[-length:]

    onehot_input_sequence = to_categorical(input_sequence, num_classes=NUM_PITCHES)
    onehot_input_sequence = onehot_input_sequence[np.newaxis, ...]

    probabilities = model.predict(onehot_input_sequence, verbose=0)[0]
    probabilities = np.maximum(probabilities, 1e-10)

    output_int = _sample_with_temperature(probabilities, temperature)

    input_sequence.append(output_int)

    output_symbol = [k for k, v in mappings.items() if v == output_int][0]

    # if output_symbol == "/":
    #   output_symbol = 'r'
    #   num_off_notes += 1
    #   if num_off_notes == 1:
    #     break

    melody.append(output_symbol)

  return melody

def save_melody(melody, step_duration=0.25, format="midi", file_name="mel.mid"):
        # create a music21 stream
        stream = m21.stream.Stream()

        start_symbol = None
        step_counter = 1

        # parse all the symbols in the melody and create note/rest objects
        for i, symbol in enumerate(melody):

            # handle case in which we have a note/rest
            if symbol != "_" or i + 1 == len(melody):

                # ensure we're dealing with note/rest beyond the first one
                if start_symbol is not None:

                    quarter_length_duration = step_duration * step_counter # 0.25 * 4 = 1

                    # handle rest
                    if start_symbol == "r":
                        m21_event = m21.note.Rest(quarterLength=quarter_length_duration)

                    # handle note
                    else:
                        m21_event = m21.note.Note(int(start_symbol), quarterLength=quarter_length_duration)

                    stream.append(m21_event)

                    # reset the step counter
                    step_counter = 1

                start_symbol = symbol

            # handle case in which we have a prolongation sign "_"
            else:
                step_counter += 1

        # write the m21 stream to a midi file
        stream.write(format, file_name)

gener_notes = generate_melody(seed, 100, SEQUENCE_LENGTH, 0.3)
save_melody(gener_notes, step_duration = 0.25, file_name="generate_melody_rl.mid")

