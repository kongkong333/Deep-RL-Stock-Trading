import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

from utils import Portfolio


class Agent(Portfolio):
    def __init__(self, state_dim, balance, is_eval=False, model_name=""):
        super().__init__(balance=balance)
        self.model_type = 'PPO'
        self.state_dim = state_dim
        self.action_dim = 3
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_ratio = 0.2
        self.update_epochs = 10
        self.batch_size = 64
        self.is_eval = is_eval
        self.memory = []
        self.optimizer = Adam(learning_rate=3e-4)
        self.model = self.create_model()
        self.model.compile(optimizer=self.optimizer)
        if is_eval:
            self.model.load_weights(f'saved_models/{model_name}.h5')
        self.tensorboard = TensorBoard(log_dir='./logs/PPO_tensorboard', update_freq=1)
        self.tensorboard.set_model(self.model)

    def create_model(self):
        states = Input(shape=(self.state_dim,))
        hidden = Dense(64, activation='tanh')(states)
        hidden = Dense(64, activation='tanh')(hidden)
        logits = Dense(self.action_dim, name='policy')(hidden)
        value = Dense(1, name='value')(hidden)
        return Model(states, [logits, value])

    def reset(self):
        self.reset_portfolio()
        self.memory.clear()

    def act(self, state, action_mask=None):
        logits, value = self.model(state, training=False)
        logits = logits.numpy()[0]
        mask = np.ones(self.action_dim, dtype=bool) if action_mask is None else action_mask
        logits[~mask] = -1e9
        probabilities = tf.nn.softmax(logits).numpy()
        action = int(np.argmax(probabilities)) if self.is_eval else int(np.random.choice(self.action_dim, p=probabilities))
        self.action_probs = probabilities
        self.last_log_prob = float(np.log(max(probabilities[action], 1e-8)))
        self.last_value = float(value.numpy()[0, 0])
        return action

    def remember(self, state, action, reward, next_state, done, action_mask=None):
        if action_mask is None:
            action_mask = np.ones(self.action_dim, dtype=bool)
        self.memory.append((state[0].copy(), int(action), float(reward), bool(done),
                            self.last_log_prob, self.last_value, np.asarray(action_mask, dtype=bool)))

    def experience_replay(self):
        states, actions, rewards, dones, old_log_probs, values, masks = map(np.asarray, zip(*self.memory))
        states = states.astype(np.float32)
        actions = actions.astype(np.int32)
        rewards = rewards.astype(np.float32)
        dones = dones.astype(np.float32)
        old_log_probs = old_log_probs.astype(np.float32)
        values = values.astype(np.float32)
        masks = masks.astype(bool)

        advantages = np.zeros_like(rewards)
        gae = 0.0
        for index in reversed(range(len(rewards))):
            next_value = 0.0 if dones[index] else (values[index + 1] if index + 1 < len(values) else 0.0)
            delta = rewards[index] + self.gamma * next_value * (1 - dones[index]) - values[index]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[index]) * gae
            advantages[index] = gae
        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        losses = []
        indices = np.arange(len(states))
        for _ in range(self.update_epochs):
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start:start + self.batch_size]
                batch_states = tf.convert_to_tensor(states[batch])
                batch_actions = tf.convert_to_tensor(actions[batch])
                batch_masks = tf.convert_to_tensor(masks[batch])
                batch_old_log_probs = tf.convert_to_tensor(old_log_probs[batch])
                batch_advantages = tf.convert_to_tensor(advantages[batch])
                batch_returns = tf.convert_to_tensor(returns[batch])

                with tf.GradientTape() as tape:
                    logits, predicted_values = self.model(batch_states, training=True)
                    logits = tf.where(batch_masks, logits, tf.cast(-1e9, logits.dtype))
                    log_probs = tf.nn.log_softmax(logits)
                    selected_log_probs = tf.reduce_sum(log_probs * tf.one_hot(batch_actions, self.action_dim), axis=1)
                    ratios = tf.exp(selected_log_probs - batch_old_log_probs)
                    unclipped = ratios * batch_advantages
                    clipped = tf.clip_by_value(ratios, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch_advantages
                    policy_loss = -tf.reduce_mean(tf.minimum(unclipped, clipped))
                    value_loss = tf.reduce_mean(tf.square(batch_returns - tf.squeeze(predicted_values, axis=1)))
                    probabilities = tf.nn.softmax(logits)
                    entropy = -tf.reduce_mean(tf.reduce_sum(probabilities * log_probs, axis=1))
                    loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

                gradients = tape.gradient(loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
                losses.append(float(loss.numpy()))

        self.memory.clear()
        return float(np.mean(losses))
