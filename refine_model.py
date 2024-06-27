import random
import numpy as np
from collections import deque
import json
import scipy
from collections import deque
import random

import tensorflow as tf
from keras.layers import Dense, LSTM, Dropout
from keras import Input, Model
from keras.optimizers import Adam
from keras.losses import SparseCategoricalCrossentropy

# Hyparameter
SEQUENCE_LENGTH = 64 # state_size = 64
NUM_PITCHES = 130 # action_size = 130

BATCH_SIZE = 64
MAX_LENGTH = 200 # Số notes được tạo trong melody
EPISODES = 10    # Số bước

# Path
DATA_PATH = "file_dataset.txt"
MAPPING_PATH = "mapping.json"
MODEL_PATH = 'model.h5'
RL_MODEL_PATH = 'model_rl.h5'

# Music-related constants.
INITIAL_MIDI_VALUE = 48
NUM_SPECIAL_EVENTS = 2
MIN_NOTE = 48  # Inclusive
MAX_NOTE = 84  # Exclusive
TRANSPOSE_TO_KEY = 0  # C Major
DEFAULT_QPM = 80.0

# SPECIAL NOTES
NO_EVENT = 128
NOTE_OFF = 129
NOTE_END = 130

# The number of half-steps in musical intervals, in order of dissonance
OCTAVE = 12
FIFTH = 7
THIRD = 4
SIXTH = 9
SECOND = 2
FOURTH = 5
SEVENTH = 11
HALFSTEP = 1

# Special intervals that have unique rewards
REST_INTERVAL = -1
HOLD_INTERVAL = -1.5
REST_INTERVAL_AFTER_THIRD_OR_FIFTH = -2
HOLD_INTERVAL_AFTER_THIRD_OR_FIFTH = -2.5
IN_KEY_THIRD = -3
IN_KEY_FIFTH = -5

# Indicate melody direction
ASCENDING = 1
DESCENDING = -1

# Indicate whether a melodic leap has been resolved or if another leap was made
LEAP_RESOLVED = 1
LEAP_DOUBLED = -1

# Music-related constants.
INITIAL_MIDI_VALUE = 48
NUM_SPECIAL_EVENTS = 2
MIN_NOTE = 48  # Inclusive
MAX_NOTE = 84  # Exclusive
TRANSPOSE_TO_KEY = 0  # C Major
DEFAULT_QPM = 80.0

# SPECIAL NOTES
NO_EVENT = 128
NOTE_OFF = 129
NOTE_END = 130

# The number of half-steps in musical intervals, in order of dissonance
OCTAVE = 12
FIFTH = 7
THIRD = 4
SIXTH = 9
SECOND = 2
FOURTH = 5
SEVENTH = 11
HALFSTEP = 1

# Special intervals that have unique rewards
REST_INTERVAL = -1
HOLD_INTERVAL = -1.5
REST_INTERVAL_AFTER_THIRD_OR_FIFTH = -2
HOLD_INTERVAL_AFTER_THIRD_OR_FIFTH = -2.5
IN_KEY_THIRD = -3
IN_KEY_FIFTH = -5

# Indicate melody direction
ASCENDING = 1
DESCENDING = -1

# Indicate whether a melodic leap has been resolved or if another leap was made
LEAP_RESOLVED = 1
LEAP_DOUBLED = -1

# Music theory constants used in defining reward functions.
C_MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11]
c_major = 0
a_minor = 9

C_MAJOR_KEY = []
C_MAJOR_TONIC = []
A_MINOR_TONIC = []

while (c_major < 128):
  C_MAJOR_TONIC.append(c_major)
  c_major += 12

while (a_minor < 128):
  A_MINOR_TONIC.append(a_minor)
  a_minor += 12

scale = C_MAJOR_SCALE

while (scale[-1] < 128):
  C_MAJOR_KEY = [*C_MAJOR_KEY, *scale]
  scale = [i+12 for i in scale]


# Help function
# Define the note names in one octave
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

def midi_to_note_name(midi_number):
    # Calculate the note and octave
    note = NOTE_NAMES[midi_number % 12]
    octave = (midi_number // 12) - 1

    # Return the note name with octave
    return f"{note}{octave}"

def print_midi_numbers(midi_numbers):
  for i in midi_numbers:
    print(f'{i}: ', midi_to_note_name(i))

def one_hot_encode(sequence, num_classes=NUM_PITCHES):
    encoded_sequence = np.zeros((len(sequence), num_classes))
    for i, value in enumerate(sequence):
        encoded_sequence[i, value] = 1
    return encoded_sequence

def autocorrelate(signal, lag=1):
  n = len(signal)
  x = np.asarray(signal) - np.mean(signal)
  c0 = np.var(signal)
  return (x[lag:] * x[:n - lag]).sum() / float(n) / c0

def generate_random_note_sequence(length=100, note_range=(48, 84), empty_prob=0.5, r_prob=0.1):
    notes = np.random.randint(note_range[0], note_range[1], length)

    note_strings = [str(note) for note in notes]

    final_sequence = []
    for i, note in enumerate(note_strings):
        random_value = np.random.rand()
        if random_value < empty_prob and i >= 1:
            final_sequence.append('_')
        elif random_value < empty_prob + r_prob and i >= 1:
            final_sequence.append('r')
        else:
            final_sequence.append(note)

    return ' '.join(final_sequence)

def convert_songs_to_int(songs, mapping_path=MAPPING_PATH):
    int_songs = []

    with open(MAPPING_PATH, "r") as fp:
      mappings = json.load(fp)

    songs = songs.split()
    for symbol in songs:
      int_songs.append(mappings[symbol])

    return int_songs

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

# def reward_from_reward_rnn_scores(action, reward_scores):
#   action_note = np.argmax(action)
#   normalization_constant = scipy.special.logsumexp(reward_scores)
#   return reward_scores[action_note] - normalization_constant

# Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        # replay memory
        self.memory = deque(maxlen=10000)

        # hyper params
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.temperature = 2.0
        self.step_update_target = 10

        # model
        self.main_network = self._build_model()
        self.target_network = self._build_model()
        self.update_target_network()

    def update_target_network(self):
        self.target_network.set_weights(self.main_network.get_weights())

    def _build_model(self):
        model = NoteRNN(output_units=self.action_size)
        model.load_weights('model.h5')
        return model

    # add experience
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # make decision
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return one_hot_encode([np.random.randint(self.action_size)], num_classes=NUM_PITCHES)
        act_values = self.main_network.predict(state, verbose=0)
        return act_values

    # get batch from buffer
    def get_batch_from_buffer(self, batch_size):
        exp_batch = random.sample(self.memory, batch_size)

        state_batch  = np.array([batch[0] for batch in exp_batch]).reshape(batch_size, self.state_size, self.action_size)
        action_batch = np.array([batch[1] for batch in exp_batch])
        reward_batch = [batch[2] for batch in exp_batch]
        next_state_batch = np.array([batch[3] for batch in exp_batch]).reshape(batch_size, self.state_size, self.action_size)
        terminal_batch = [batch[4] for batch in exp_batch]
        return state_batch, action_batch, reward_batch, next_state_batch, terminal_batch

    # train
    def train_main_network(self, batch_size=64):
        state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = self.get_batch_from_buffer(batch_size)

        # Get Q-values of the current state s
        q_values = self.main_network.predict(state_batch, verbose=0)

        # Get Max Q-values of the next state s'
        next_q_values = self.target_network.predict(next_state_batch, verbose=0)
        max_next_q = np.amax(next_q_values, axis=1)

        for i in range(batch_size):
            new_q_values = reward_batch[i] if terminal_batch[i] else reward_batch[i] + self.gamma * max_next_q[i]
            action_note = np.argmax(action_batch[i])
            q_values[i][action_note] = new_q_values

        q_actions = np.argmax(q_values, axis=1)

        self.main_network.fit(state_batch, q_actions, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Environment
class MusicEnv:
    def __init__(self, max_length=100):
        self.max_length = max_length
        self.num_notes_in_melody = SEQUENCE_LENGTH

        self.beat = 0
        self.sequence_input = None
        self.composition = []

        self.composition_direction = 0
        self.steps_since_last_leap = 0
        self.leapt_from = None

        self.reset()

    def reset(self):
        random_note = generate_random_note_sequence(length=64)

        self.sequence_input = one_hot_encode(convert_songs_to_int(random_note), num_classes=130)
        self.composition = []
        self.composition_direction = 0
        self.beat = 0
        self.leapt_from = None
        self.steps_since_last_leap = 0
        return self.sequence_input, self.composition

    def step(self, action):
        action_note = np.argmax(action)
        self.beat += 1
        self.composition.append(action_note)
        self.sequence_input =  np.append(self.sequence_input, action, axis=0)[1:]

        if len(self.composition) > self.max_length or action_note == NOTE_END:
            done = True
            reward = self.reward_music_theory(action)
        else:
            done = False
            reward = 0

        return self.sequence_input, reward, done

    # reward function

    def reward_key(self, action, penalty_amount=-1.0, key=None):
      if key is None:
        key = C_MAJOR_KEY

      reward = 0

      action_note = np.argmax(action)
      if action_note not in key:
        reward = penalty_amount

      return reward

    def reward_tonic(self, action, tonic_note=C_MAJOR_TONIC, reward_amount=3.0):
      action_note = np.argmax(action)
      first_note_of_final_bar = self.num_notes_in_melody - 4

      if self.beat == 0 or self.beat == first_note_of_final_bar:
        if action_note == tonic_note:
          return reward_amount
      elif self.beat == first_note_of_final_bar + 1:
        if action_note == NO_EVENT:
          return reward_amount
      elif self.beat > first_note_of_final_bar + 1:
        if action_note in (NO_EVENT, NOTE_OFF):
          return reward_amount
      return 0.0

    def reward_non_repeating(self, action):
      penalty = self.reward_penalize_repeating(action)
      if penalty >= 0:
        return .1
      else:
        return 0.0

    def detect_repeating_notes(self, action_note):
      num_repeated = 0
      contains_held_notes = False
      contains_breaks = False

      # Note that the current action yas not yet been added to the composition
      for i in range(len(self.composition)-1, -1, -1):
        if self.composition[i] == action_note:
          num_repeated += 1
        elif self.composition[i] == NOTE_OFF:
          contains_breaks = True
        elif self.composition[i] == NO_EVENT:
          contains_held_notes = True
        else:
          break

      if action_note == NOTE_OFF and num_repeated > 1:
        return True
      elif not contains_held_notes and not contains_breaks:
        if num_repeated > 4:
          return True
      elif contains_held_notes or contains_breaks:
        if num_repeated > 6:
          return True
      else:
        if num_repeated > 8:
          return True

      return False

    def reward_penalize_repeating(self, action, penalty_amount=-100.0):
      action_note = np.argmax(action)
      is_repeating = self.detect_repeating_notes(action_note)
      if is_repeating:
        return penalty_amount
      else:
        return 0.0

    def reward_penalize_autocorrelation(self, action, penalty_weight=3.0):
      composition = self.composition + [np.argmax(action)]
      lags = [1, 2, 3]
      sum_penalty = 0
      for lag in lags:
        coeff = autocorrelate(composition, lag=lag)
        if not np.isnan(coeff):
          if np.abs(coeff) > 0.15:
            sum_penalty += np.abs(coeff) * penalty_weight
      return -sum_penalty

    def detect_last_motif(self, composition=None, bar_length=8):
      if composition is None:
        composition = self.composition

      if len(composition) < bar_length:
        return None, 0

      last_bar = composition[-bar_length:]

      actual_notes = [a for a in last_bar if a not in (NO_EVENT, NOTE_OFF)]
      num_unique_notes = len(set(actual_notes))
      if num_unique_notes >= 3:
        return last_bar, num_unique_notes
      else:
        return None, num_unique_notes

    def reward_motif(self, action, reward_amount=3.0):
      composition = self.composition + [np.argmax(action)]
      motif, num_notes_in_motif = self.detect_last_motif(composition=composition)
      if motif is not None:
        motif_complexity_bonus = max((num_notes_in_motif - 3)*.3, 0)
        return reward_amount + motif_complexity_bonus
      else:
        return 0.0

    def detect_repeated_motif(self, action, bar_length=8):
      composition = self.composition + [np.argmax(action)]
      if len(composition) < bar_length:
        return False, None

      motif, _ = self.detect_last_motif(composition=composition, bar_length=bar_length)
      if motif is None:
        return False, None

      prev_composition = self.composition[:-(bar_length-1)]

      # Check if the motif is in the previous composition.
      for i in range(len(prev_composition) - len(motif) + 1):
        for j in range(len(motif)):
          if prev_composition[i + j] != motif[j]:
            break
        else:
          return True, motif
      return False, None

    def reward_repeated_motif(self, action, bar_length=8, reward_amount=4.0):
      is_repeated, motif = self.detect_repeated_motif(action, bar_length)
      if is_repeated:
        actual_notes = [a for a in motif if a not in (NO_EVENT, NOTE_OFF)]
        num_notes_in_motif = len(set(actual_notes))
        motif_complexity_bonus = max(num_notes_in_motif - 3, 0)
        return reward_amount + motif_complexity_bonus
      else:
        return 0.0

    def detect_sequential_interval(self, action, key=None):
      if not self.composition:
        return 0, None, None

      prev_note = self.composition[-1]
      action_note = np.argmax(action)

      c_major = False
      if key is None:
        key = C_MAJOR_KEY
        c_notes = [2, 14, 26]
        g_notes = [9, 21, 33]
        e_notes = [6, 18, 30]
        c_major = True
        tonic_notes = [2, 14, 26]
        fifth_notes = [9, 21, 33]

      # get rid of non-notes in prev_note
      prev_note_index = len(self.composition) - 1
      while prev_note in (NO_EVENT, NOTE_OFF) and prev_note_index >= 0:
        prev_note = self.composition[prev_note_index]
        prev_note_index -= 1
      if prev_note in (NOTE_OFF, NO_EVENT):
        return 0, action_note, prev_note

      # get rid of non-notes in action_note
      if action_note == NO_EVENT:
        if prev_note in tonic_notes or prev_note in fifth_notes:
          return (HOLD_INTERVAL_AFTER_THIRD_OR_FIFTH, action_note, prev_note)
        else:
          return HOLD_INTERVAL, action_note, prev_note
      elif action_note == NOTE_OFF:
        if prev_note in tonic_notes or prev_note in fifth_notes:
          return (REST_INTERVAL_AFTER_THIRD_OR_FIFTH,
                  action_note, prev_note)
        else:
          return REST_INTERVAL, action_note, prev_note

      interval = abs(action_note - prev_note)

      if c_major and interval == FIFTH and (
          prev_note in c_notes or prev_note in g_notes):
        return IN_KEY_FIFTH, action_note, prev_note
      if c_major and interval == THIRD and (
          prev_note in c_notes or prev_note in e_notes):
        return IN_KEY_THIRD, action_note, prev_note

      return interval, action_note, prev_note

    def reward_preferred_intervals(self, action, scaler=5.0, key=None):
      interval, _, _ = self.detect_sequential_interval(action, key)

      if interval == 0:  # either no interval or involving uninteresting rests
        return 0.0

      reward = 0.0

      # rests can be good
      if interval == REST_INTERVAL:
        reward = 0.05
      if interval == HOLD_INTERVAL:
        reward = 0.075
      if interval == REST_INTERVAL_AFTER_THIRD_OR_FIFTH:
        reward = 0.15
      if interval == HOLD_INTERVAL_AFTER_THIRD_OR_FIFTH:
        reward = 0.3

      # large leaps and awkward intervals bad
      if interval == SEVENTH:
        reward = -0.3
      if interval > OCTAVE:
        reward = -1.0

      # common major intervals are good
      if interval == IN_KEY_FIFTH:
        reward = 0.1
      if interval == IN_KEY_THIRD:
        reward = 0.15

      # smaller steps are generally preferred
      if interval == THIRD:
        reward = 0.09
      if interval == SECOND:
        reward = 0.08
      if interval == FOURTH:
        reward = 0.07

      # larger leaps not as good, especially if not in key
      if interval == SIXTH:
        reward = 0.05
      if interval == FIFTH:
        reward = 0.02

      return reward * scaler

    def detect_high_unique(self, composition):
      max_note = max(composition)
      return list(composition).count(max_note) == 1

    def detect_low_unique(self, composition):
      no_special_events = [x for x in composition if x not in (NO_EVENT, NOTE_OFF)]
      if no_special_events:
        min_note = min(no_special_events)
        if list(composition).count(min_note) == 1:
          return True
      return False

    def reward_high_low_unique(self, action, reward_amount=3.0):
      if len(self.composition) + 1 != self.num_notes_in_melody:
        return 0.0

      composition = np.array(self.composition)
      composition = np.append(composition, np.argmax(action))

      reward = 0.0

      if self.detect_high_unique(composition):
        reward += reward_amount

      if self.detect_low_unique(composition):
        reward += reward_amount

      return reward

    def detect_leap_up_back(self, action, steps_between_leaps=6):

      if not self.composition:
        return 0

      outcome = 0

      interval, action_note, prev_note = self.detect_sequential_interval(action)

      if action_note in (NOTE_OFF, NO_EVENT):
        self.steps_since_last_leap += 1
        return 0

      # detect if leap
      if interval >= FIFTH or interval == IN_KEY_FIFTH:
        if action_note > prev_note:
          leap_direction = ASCENDING
        else:
          leap_direction = DESCENDING

        # there was already an unresolved leap
        if self.composition_direction != 0:
          if self.composition_direction != leap_direction:
            if self.steps_since_last_leap > steps_between_leaps:
              outcome = LEAP_RESOLVED
            self.composition_direction = 0
            self.leapt_from = None
          else:
            outcome = LEAP_DOUBLED

        # the composition had no previous leaps
        else:
          self.composition_direction = leap_direction
          self.leapt_from = prev_note

        self.steps_since_last_leap = 0

      # there is no leap
      else:
        self.steps_since_last_leap += 1

        # If there was a leap before, check if composition has gradually returned
        # This could be changed by requiring you to only go a 5th back in the
        # opposite direction of the leap.
        if (self.composition_direction == ASCENDING and
                action_note <= self.leapt_from) or (self.composition_direction == DESCENDING and
                action_note >= self.leapt_from):
              outcome = LEAP_RESOLVED
              self.composition_direction = 0
              self.leapt_from = None

      return outcome

    def reward_leap_up_back(self, action, resolving_leap_bonus=5.0, leaping_twice_punishment=-5.0):
      leap_outcome = self.detect_leap_up_back(action)
      if leap_outcome == LEAP_RESOLVED:
        return resolving_leap_bonus
      elif leap_outcome == LEAP_DOUBLED:
        return leaping_twice_punishment
      else:
        return 0.0

      # def reward_music_theory(self, action):
      #   reward = (
      #       self.reward_key(action) +
      #       self.reward_tonic(action) +
      #       self.reward_penalize_repeating(action) +
      #       self.reward_penalize_autocorrelation(action) +
      #       self.reward_motif(action) +
      #       self.reward_repeated_motif(action) +
      #       self.reward_preferred_intervals(action) +
      #       self.reward_leap_up_back(action) +
      #       self.reward_high_low_unique(action)
      #   )

      #   return reward

    def reward_music_theory(self, action):
      reward_key = self.reward_key(action)
      reward_tonic = self.reward_tonic(action)
      reward_penalize_repeating = self.reward_penalize_repeating(action)
      reward_penalize_autocorrelation = self.reward_penalize_autocorrelation(action)
      reward_motif = self.reward_motif(action)
      reward_repeated_motif = self.reward_repeated_motif(action)
      reward_preferred_intervals = self.reward_preferred_intervals(action)
      reward_leap_up_back = self.reward_leap_up_back(action)
      reward_high_low_unique = self.reward_high_low_unique(action)

      # print('\n---------------------------------')
      # print('Key: ', reward_key)
      # print('Tonic: ', reward_tonic)
      # print('Penalize Repeating: ', reward_penalize_repeating)
      # print('Penalize Autocorrelation: ', reward_penalize_autocorrelation)
      # print('Motif: ', reward_motif)
      # print('Repeated Motif: ', reward_repeated_motif)
      # print('Preferred Intervals: ', reward_preferred_intervals)
      # print('Leap Up/Back: ', reward_leap_up_back)
      # print('High/Low Unique: ', reward_high_low_unique)

      reward = (
          reward_key +
          reward_tonic +
          reward_penalize_repeating +
          reward_penalize_autocorrelation +
          reward_motif +
          reward_repeated_motif +
          reward_preferred_intervals +
          reward_leap_up_back +
          reward_high_low_unique
      )

      return reward

# Refine model with Deep Q-Network
state_size = SEQUENCE_LENGTH
action_size = NUM_PITCHES
batch_size = BATCH_SIZE
max_length = MAX_LENGTH

# Env
env = MusicEnv(max_length=MAX_LENGTH)

# Agent
agent = DQNAgent(state_size, action_size)

# Loop
episodes =  EPISODES
total_time_step = 0

# Check
new_state = []

for e in range(episodes):
    ep_reward = 0

    state, _ = env.reset()
    state = np.reshape(state, [1, state_size, action_size])
    for time in range(500):
        total_time_step += 1

        # Update target network
        if total_time_step % agent.step_update_target == 0:
            # print(f"update target network at {total_time_step}")
            agent.update_target_network()

        # Train main
        action = agent.act(state)

        # Update state
        next_state, reward, done = env.step(action)
        next_state = np.reshape(next_state[-state_size:], [1, state_size, action_size])

        # Save experience
        agent.remember(state, action, reward, next_state, done)
        state = next_state

        ep_reward += reward

        if done:
            print(f"episode: {e}/{episodes}, score: {time}, e: {agent.epsilon:.2}")
            break

    new_state.append(env.composition)

    if len(agent.memory) > batch_size:
        agent.train_main_network(batch_size)

# Save model
agent.target_network.save_weights(RL_MODEL_PATH)