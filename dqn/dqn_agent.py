import tensorflow as tf
from tensorflow.keras import layers

class DQNAgent:
    def __init__(self, n_states, n_actions, batch_size, input_size):
        self._n_states = n_states
        self._n_actions = n_actions
        self._batch_size = batch_size
        self._input_size = input_size

        # Define State and Action placeholders
        self._states = None
        self._actions = None

        # Output operations
        self._logits = None
        self._optimizer = None
        self._variable_initializer = None

        # Model setup
        self._define_network()

    def _define_network(self):
        model = tf.keras.Sequential()
        model.add(layers.Input())
        model.add(layers.Conv2D(filters=16, kernel_size=(8, 8), strides=4, activation='relu'))
        model.add(layers.Conv2D(filters=32, kernel_size=(4, 4), strides=2, activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(units=256, activation='relu'))
        model.add(layers.Dense(self._n_actions))
        return model
