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

# Params
DATA_PATH = "file_dataset.txt"
MAPPING_PATH = "mapping.json"

SEQUENCE_LENGTH = 64
NUM_PITCHES = 130

EPOCHS = 3
BATCH_SIZE = 64

ACCEPTABLE_DURATIONS = [0.25,  0.5,  0.75, 1.0, 1.5, 2, 3, 4]

# Data generator
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

    int_songs = [song for song in int_songs if song != 130]

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

# Generator
OUTPUT_UNITS = NUM_PITCHES

with open(DATA_PATH, "r") as fp:
  songs = fp.read()

int_songs = convert_songs_to_int(songs)
steps_per_epoch = (len(int_songs) - SEQUENCE_LENGTH) // BATCH_SIZE


generator = generate_sequences_generator(SEQUENCE_LENGTH, BATCH_SIZE)

# Train
model = NoteRNN(OUTPUT_UNITS)
model.fit(generator, steps_per_epoch=steps_per_epoch, epochs=EPOCHS)

# Save model
model.save_weights('model.h5')