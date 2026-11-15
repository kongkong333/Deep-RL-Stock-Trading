import random
from collections import deque

import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

from utils import Portfolio


# reference:
# https://arxiv.org/pdf/1312.5602.pdf
class Agent(Portfolio):
    def __init__(self, state_dim, balance, is_eval=False, model_name=""):
        super().__init__(balance=balance)
        self.model_type = 'DQN'
        self.state_dim = state_dim
        self.action_dim = 3  # hold, buy, sell
        self.memory = deque(maxlen=10000)
        self.buffer_size = 64
        self.learning_starts = 100
        self.train_freq = 1
        self.tau = 0.001

        self.gamma = 0.95
        self.epsilon = 1.0  # initial exploration rate
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9995
        self.is_eval = is_eval
        self.model = load_model('saved_models/{}.h5'.format(model_name)) if is_eval else self.create_model()
        self.model_target = self.create_model()
        self.model_target.set_weights(self.model.get_weights())

        self.tensorboard = TensorBoard(log_dir='./logs/DQN_tensorboard', update_freq=90)
        self.tensorboard.set_model(self.model)

    def create_model(self):
        model = Sequential()
        model.add(Dense(units=64, input_dim=self.state_dim, activation='relu'))
        model.add(Dense(units=32, activation='relu'))
        model.add(Dense(units=8, activation='relu'))
        model.add(Dense(self.action_dim))
        model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
        return model

    def reset(self):
        self.reset_portfolio()

    def remember(self, state, actions, reward, next_state, done, next_action_mask=None):
        if next_action_mask is None:
            next_action_mask = np.ones(self.action_dim, dtype=bool)
        self.memory.append((state, int(actions), reward, next_state, done, next_action_mask))

    def act(self, state, action_mask=None):
        if action_mask is None:
            action_mask = np.ones(self.action_dim, dtype=bool)
        valid_actions = np.flatnonzero(action_mask)
        if not self.is_eval and np.random.rand() <= self.epsilon:
            return int(np.random.choice(valid_actions))
        options = self.model.predict(state, verbose=0)[0]
        options[~action_mask] = -np.inf
        return int(np.argmax(options))

    def experience_replay(self):
        # Sample individual experiences uniformly, regardless of their order or episode.
        mini_batch = random.sample(self.memory, self.buffer_size)
        states = np.vstack([experience[0] for experience in mini_batch])
        actions = np.asarray([experience[1] for experience in mini_batch], dtype=np.int64)
        rewards = np.asarray([experience[2] for experience in mini_batch], dtype=np.float32)
        next_states = np.vstack([experience[3] for experience in mini_batch])
        dones = np.asarray([experience[4] for experience in mini_batch], dtype=np.float32)
        next_masks = np.vstack([experience[5] for experience in mini_batch])

        targets = self.model.predict(states, verbose=0)
        next_q_values = self.model_target.predict(next_states, verbose=0)
        next_q_values[~next_masks] = -np.inf
        next_q_values = next_q_values.max(axis=1)
        target_values = rewards + self.gamma * (1.0 - dones) * next_q_values
        chosen_actions = actions
        targets[np.arange(len(mini_batch)), chosen_actions] = target_values

        history = self.model.fit(states, targets, epochs=1, verbose=0)

        target_weights = self.model_target.get_weights()
        model_weights = self.model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = self.tau * model_weights[i] + (1 - self.tau) * target_weights[i]
        self.model_target.set_weights(target_weights)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return history.history['loss'][0]
