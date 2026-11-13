import random
from collections import deque

import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

from utils import Portfolio


# references:
# https://arxiv.org/pdf/1802.09477.pdf
# https://arxiv.org/pdf/1509.06461.pdf
# https://papers.nips.cc/paper/3964-double-q-learning.pdf
class Agent(Portfolio):
    def __init__(self, state_dim, balance, is_eval=False, model_name=""):
        super().__init__(balance=balance)
        self.model_type = 'DDQN'
        self.state_dim = state_dim
        self.action_dim = 3  # hold, buy, sell
        self.memory = deque(maxlen=10000)
        self.buffer_size = 64
        self.learning_starts = 100
        self.train_freq = 4

        self.tau = 0.0001
        self.gamma = 0.95
        self.epsilon = 1.0  # initial exploration rate
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9995
        self.is_eval = is_eval

        self.model = load_model(f'saved_models/{model_name}.h5') if is_eval else self.create_model()
        self.model_target = self.create_model()
        self.model_target.set_weights(self.model.get_weights()) # hard copy model parameters to target model parameters

        self.tensorboard = TensorBoard(log_dir='./logs/DDQN_tensorboard', update_freq=90)
        self.tensorboard.set_model(self.model)

    def update_model_target(self):
        model_weights = self.model.get_weights()
        model_target_weights = self.model_target.get_weights()
        for i in range(len(model_weights)):
            model_target_weights[i] = self.tau * model_weights[i] + (1 - self.tau) * model_target_weights[i]
        self.model_target.set_weights(model_target_weights)

    def create_model(self):
        model = Sequential()
        model.add(Dense(units=64, input_dim=self.state_dim, activation='relu'))
        model.add(Dense(units=32, activation='relu'))
        model.add(Dense(units=8, activation='relu'))
        model.add(Dense(self.action_dim))
        model.compile(loss='mse', optimizer=Adam(lr=0.001))
        return model

    def reset(self):
        self.reset_portfolio()

    def remember(self, state, action, reward, next_state, done, next_action_mask=None):
        if next_action_mask is None:
            next_action_mask = np.ones(self.action_dim, dtype=bool)
        self.memory.append((state, int(action), reward, next_state, done, next_action_mask))

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
        # Choose a random start, then keep the batch chronological and within one episode.
        memories = list(self.memory)
        valid_starts = [
            start for start in range(len(memories) - self.buffer_size + 1)
            if not any(item[4] for item in memories[start:start + self.buffer_size - 1])
        ]
        start = random.choice(valid_starts)
        mini_batch = memories[start:start + self.buffer_size]

        states = np.vstack([item[0] for item in mini_batch])
        actions = np.asarray([item[1] for item in mini_batch], dtype=np.int64)
        rewards = np.asarray([item[2] for item in mini_batch], dtype=np.float32)
        next_states = np.vstack([item[3] for item in mini_batch])
        dones = np.asarray([item[4] for item in mini_batch], dtype=np.float32)
        next_masks = np.vstack([item[5] for item in mini_batch])

        targets = self.model.predict(states, verbose=0)
        next_q_online = self.model.predict(next_states, verbose=0)
        next_q_online[~next_masks] = -np.inf
        next_actions = next_q_online.argmax(axis=1)
        next_q_values = self.model_target.predict(next_states, verbose=0)
        target_values = rewards + (1 - dones) * self.gamma * next_q_values[np.arange(len(mini_batch)), next_actions]
        targets[np.arange(len(mini_batch)), actions] = target_values
        history = self.model.fit(states, targets, epochs=1, verbose=0)
        self.update_model_target()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return history.history['loss'][0]
