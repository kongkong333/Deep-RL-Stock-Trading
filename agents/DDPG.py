import random
from collections import deque

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam

from utils import Portfolio

# Tensorflow GPU configuration
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)
tf.compat.v1.disable_eager_execution()


HIDDEN1_UNITS = 24
HIDDEN2_UNITS = 48
HIDDEN3_UNITS = 24


# reference: 
# https://arxiv.org/pdf/1509.02971.pdf
class ActorNetwork:
    def __init__(self, sess, state_size, action_dim, buffer_size, tau, learning_rate, is_eval=False, model_name=""):
        self.sess = sess
        self.tau = tau
        self.learning_rate = learning_rate
        self.action_dim = action_dim
        if is_eval == True:
            self.model, self.states = self.create_actor_network(state_size, action_dim)
            self.model.load_weights('saved_models/{}_actor.h5'.format(model_name))
        else:
            self.model, self.states = self.create_actor_network(state_size, action_dim)
            self.model_target, self.target_state = self.create_actor_network(state_size, action_dim)
            self.model_target.set_weights(self.model.get_weights()) # hard copy model parameters to target model parameters

            self.action_gradient = tf.compat.v1.placeholder(tf.float32, [None, action_dim])
            # chain rule: ∂a/∂θ * ∂Q(s,a)/∂a (action_gradients); minus sign for gradient descent; 1/buffer_size for mean value
            self.sampled_policy_grad = tf.gradients(self.model.output/buffer_size, self.model.trainable_weights, -self.action_gradient)
            self.update_actor_policy = Adam(learning_rate=learning_rate).apply_gradients(zip(self.sampled_policy_grad, self.model.trainable_weights))

    def train(self, states_batch, action_grads_batch):
        self.sess.run(self.update_actor_policy, feed_dict={self.states: states_batch, self.action_gradient: action_grads_batch})

    def train_target(self):
        actor_weights = self.model.get_weights()
        actor_target_weights = self.model_target.get_weights()
        for i in range(len(actor_weights)):
            actor_target_weights[i] = self.tau * actor_weights[i] + (1 - self.tau) * actor_target_weights[i]
        self.model_target.set_weights(actor_target_weights)

    def create_actor_network(self, state_size, action_dim):
        states = Input(shape=[state_size])
        h0 = Dense(HIDDEN1_UNITS, activation='relu')(states)
        h1 = Dense(HIDDEN2_UNITS, activation='relu')(h0)
        h2 = Dense(HIDDEN3_UNITS, activation='relu')(h1)
        actions = Dense(self.action_dim, activation='softmax')(h2)
        model = Model(inputs=states, outputs=actions)
        return model, states


class CriticNetwork:
    def __init__(self, sess, state_size, action_dim, tau, learning_rate, is_eval=False, model_name=""):
        self.sess = sess
        self.tau = tau
        self.learning_rate = learning_rate
        self.action_dim = action_dim
        if is_eval == True:
            self.model, self.actions, self.states = self.create_critic_network(state_size, action_dim)
            self.model.load_weights('saved_models/{}_critic.h5'.format(model_name))
        else:
            self.model, self.actions, self.states = self.create_critic_network(state_size, action_dim)
            self.model_target, self.target_action, self.target_state = self.create_critic_network(state_size, action_dim)
            self.action_grads = tf.gradients(self.model.output, self.actions)

    def gradients(self, states_batch, actions_batch):
        return self.sess.run(self.action_grads, feed_dict={self.states: states_batch, self.actions: actions_batch})[0]

    def train_target(self):
        critic_weights = self.model.get_weights()
        critic_target_weights = self.model_target.get_weights()
        for i in range(len(critic_weights)):
            critic_target_weights[i] = self.tau * critic_weights[i] + (1 - self.tau) * critic_target_weights[i]
        self.model_target.set_weights(critic_target_weights)

    def create_critic_network(self, state_size, action_dim):
        states = Input(shape=[state_size])
        actions = Input(shape=[action_dim])
        h0 = Concatenate()([states, actions])
        h1 = Dense(HIDDEN1_UNITS, activation='relu')(h0)
        h2 = Dense(HIDDEN2_UNITS, activation='relu')(h1)
        h3 = Dense(HIDDEN3_UNITS, activation='relu')(h2)
        Q = Dense(1)(h3)
        model = Model(inputs=[states, actions], outputs=Q)
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate, decay=1e-6))
        return model, actions, states


# reference: https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py
class OUNoise:
    def __init__(self, action_dim, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000):
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_dim
        self.reset()

    def reset(self):
        self.states = np.ones(self.action_dim) * self.mu

    def evolve_state(self):
        x = self.states
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.states = x + dx
        return self.states

    def get_actions(self, actions, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(actions + ou_state, 0, 1)


class Agent(Portfolio):
    def __init__(self, state_dim, balance, is_eval=False, model_name=""):
        super().__init__(balance=balance)
        self.model_type = 'DDPG'
        self.state_dim = state_dim
        self.action_dim = 3  # hold, buy, sell
        self.memory = deque(maxlen=100)
        self.buffer_size = 90

        self.gamma = 0.95 # discount factor
        self.is_eval = is_eval
        self.noise = OUNoise(self.action_dim, min_sigma=0.05)
        tau = 0.001  # Target network hyperparameter
        learning_rate_actor = 0.001  # learning rate for Actor network
        learning_rate_critic = 0.001  # learning rate for Critic network

        self.actor = ActorNetwork(sess, state_dim, self.action_dim, self.buffer_size, tau, learning_rate_actor, is_eval, model_name)
        self.critic = CriticNetwork(sess, state_dim, self.action_dim, tau, learning_rate_critic)

        self.tensorboard = tf.keras.callbacks.TensorBoard(log_dir='./logs/DDPG_tensorboard', update_freq=90)
        self.tensorboard.set_model(self.critic.model)

    def reset(self):
        self.reset_portfolio()
        self.noise.reset()

    def remember(self, state, actions, reward, next_state, done):
        self.memory.append((state, actions, reward, next_state, done))

    def act(self, state, t):
        actions = self.actor.model.predict(state)[0]
        if not self.is_eval:
            return self.noise.get_actions(actions, t)
        return actions

    def experience_replay(self):
        # sample random buffer_size long memory
        mini_batch = random.sample(self.memory, self.buffer_size)

        states_batch = np.vstack([item[0] for item in mini_batch])
        actions_batch = np.vstack([item[1] for item in mini_batch])
        rewards = np.asarray([item[2] for item in mini_batch], dtype=np.float32).reshape(-1, 1)
        next_states = np.vstack([item[3] for item in mini_batch])
        dones = np.asarray([item[4] for item in mini_batch], dtype=np.float32).reshape(-1, 1)
        next_actions = self.actor.model_target.predict(next_states, verbose=0)
        next_q_values = self.critic.model_target.predict([next_states, next_actions], verbose=0)
        y_batch = rewards + self.gamma * (1 - dones) * next_q_values
        
        # update critic by minimizing the loss
        loss = self.critic.model.train_on_batch([states_batch, actions_batch], y_batch)

        # update actor using the sampled policy gradients
        action_grads_batch = self.critic.gradients(states_batch, self.actor.model.predict(states_batch, verbose=0))
        self.actor.train(states_batch, action_grads_batch)
        
        # update target networks
        self.actor.train_target()
        self.critic.train_target()
        return loss
