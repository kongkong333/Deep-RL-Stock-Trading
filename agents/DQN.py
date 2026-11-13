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
        self.memory = deque(maxlen=100)
        self.buffer_size = 60
        self.tau = 0.001

        self.gamma = 0.95
        self.epsilon = 1.0  # initial exploration rate
        self.epsilon_min = 0.01  # minimum exploration rate
        self.epsilon_decay = 0.995 # decrease exploration rate as the agent becomes good at trading
        self.is_eval = is_eval
        self.model = load_model('saved_models/{}.h5'.format(model_name)) if is_eval else self.model()
        self.model_target = self.model()
        self.model_target.set_weights(self.model.get_weights())

        self.tensorboard = TensorBoard(log_dir='./logs/DQN_tensorboard', update_freq=90)
        self.tensorboard.set_model(self.model)

    def model(self):
        model = Sequential()
        model.add(Dense(units=64, input_dim=self.state_dim, activation='relu'))
        model.add(Dense(units=32, activation='relu'))
        model.add(Dense(units=8, activation='relu'))
        model.add(Dense(self.action_dim))
        model.compile(loss='mse', optimizer=Adam(lr=0.01))
        return model

    def reset(self):
        self.reset_portfolio()
        self.epsilon = 1.0 # reset exploration rate

    def remember(self, state, actions, reward, next_state, done):
        self.memory.append((state, int(actions), reward, next_state, done))

    def act(self, state):
        if not self.is_eval and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim)
        options = self.model.predict(state)
        return np.argmax(options[0])

    def experience_replay(self):
        mini_batch = random.sample(self.memory, self.buffer_size)
        states = np.vstack([experience[0] for experience in mini_batch])
        actions = np.asarray([experience[1] for experience in mini_batch], dtype=np.int64)
        rewards = np.asarray([experience[2] for experience in mini_batch], dtype=np.float32)
        next_states = np.vstack([experience[3] for experience in mini_batch])
        dones = np.asarray([experience[4] for experience in mini_batch], dtype=np.float32)

        targets = self.model.predict(states, verbose=0)
        next_q_values = self.model_target.predict(next_states, verbose=0).max(axis=1)
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
