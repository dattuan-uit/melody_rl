import os
import json
import glob
import pathlib
import pandas as pd
import numpy as np
import music21 as m21

from google.colab import drive
drive.mount('/content/drive')


DATASET_PATH = '/content/drive/MyDrive/deutschl/erk'
FINAL_DATASET = "file_dataset.txt"
MAPPING_PATH = "mapping.json"
SAVE_DIR = "dataset"

SEQUENCE_LENGTH = 64
NUM_PITCHES = 131

ACCEPTABLE_DURATIONS = [0.25,  0.5,  0.75, 1.0, 1.5, 2, 3, 4]

def load_songs_in_kern(dataset_path):
    songs = []

    for path, subdirs, files in os.walk(dataset_path):
      for file in files:
        if file[-3:] == "krn":
        # if file[-4:] == "midi":
          song = m21.converter.parse(os.path.join(path, file))
          songs.append(song)

    return songs

def has_acceptable_durations(song, acceptable_durations):
  for note in song.flatten().notesAndRests:
    if note.duration.quarterLength not in acceptable_durations:
      return False

  return True

def transpose(song):
    # get key from the song
    parts = song.getElementsByClass(m21.stream.Part)
    measures_part0 = parts[0].getElementsByClass(m21.stream.Measure)
    key = measures_part0[0][4]

    # estimate key using music21
    if not isinstance(key, m21.key.Key):
      key = song.analyze("key")

    # get interval for transposition. E.g., Bmaj -> Cmaj
    if key.mode == "major":
      interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("C"))
    elif key.mode == "minor":
      interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("A"))

    # transpose song by calculated interval
    tranposed_song = song.transpose(interval)
    return tranposed_song

def encode_song(song, time_step=0.25):
  encoded_song = []

  for event in song.flatten().notesAndRests:
    if isinstance(event, m21.note.Note):
      symbol = event.pitch.midi # 60
    elif isinstance(event, m21.note.Rest):
      symbol = "r"

    # convert the note/rest into time series notation
    steps = int(event.duration.quarterLength / time_step)
    for step in range(steps):
      if step == 0:
        encoded_song.append(symbol)
      else:
        encoded_song.append("_")

  # cast encoded song to str
  encoded_song = " ".join(map(str, encoded_song))
  return encoded_song

def preprocess(dataset_path):
    # load folk songs
    print("Loading songs...")
    songs = load_songs_in_kern(dataset_path)
    print(f"Loaded {len(songs)} songs.")

    for i, song in enumerate(songs):
      # filter out songs that have non-acceptable durations
      if not has_acceptable_durations(song, ACCEPTABLE_DURATIONS):
        continue

      # transpose songs to Cmaj/Amin
      song = transpose(song)

      # encode songs with music time series representation
      encoded_song = encode_song(song)

      # save songs to text file
      save_path = os.path.join(SAVE_DIR, f'{i}.txt')
      with open(save_path, "w") as fp:
        fp.write(encoded_song)

def load(file_path):
  with open(file_path, "r") as fp:
    song = fp.read()
  return song

def create_single_file_dataset(dataset_path, file_dataset_path, sequence_length):
    new_song_delimiter = "/ " * sequence_length
    songs = ""

    # load encoded songs and add delimiters
    for path, _, files in os.walk(dataset_path):
      for file in files:
        file_path = os.path.join(path, file)
        song = load(file_path)
        songs = songs + song + " " + new_song_delimiter

    # remove empty space from last character of string
    songs = songs[:-1]

    with open(file_dataset_path, "w") as fp:
      fp.write(songs)

    return songs

def create_note_mapping(file_name):
    # Create a dictionary to hold the note mappings
    note_mapping = {}

    # Add special events
    note_mapping['_'] = 128
    note_mapping['r'] = 129
    note_mapping['/'] = 130

    # Add MIDI notes from 48 (C3) to 83 (B5)
    # midi_note_start = 48
    # midi_note_end = 84
    for midi_note in range(0, 128):
      note_mapping[midi_note] = midi_note

    with open(file_name, 'w') as json_file:
      json.dump(note_mapping, json_file, indent=4)

def create_mapping(songs, mapping_path):
    mappings = {}

    songs = songs.split()
    vocabulary = list(set(songs))

    for i, symbol in enumerate(vocabulary):
        mappings[symbol] = i

    with open(mapping_path, "w") as fp:
        json.dump(mappings, fp, indent=4)


preprocess(DATASET_PATH)
songs = create_single_file_dataset(SAVE_DIR, FINAL_DATASET, SEQUENCE_LENGTH)
create_note_mapping(MAPPING_PATH)