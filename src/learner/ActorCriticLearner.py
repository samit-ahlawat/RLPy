from __future__ import absolute_import, print_function

import numpy as np
import pandas as pd

import src.lib.Emulator as Emulator
from src.lib.Episode import Episode
from src.lib.ReinforcementLearner import ReinforcementLearner
import tensorflow as tf
from typing import Tuple, Sequence


class CombinedACNetwork(tf.keras.Model):
    """ Combined actor critic network """
    def __init__(self, num_actions, num_hidden_units, emulator, discount_factor, optimizer, max_steps_per_episode):
        super(CombinedACNetwork, self).__init__()
        assert isinstance(emulator, Emulator.AIGymEmulator)
        self.commonLayers = [tf.keras.layers.Dense(num_hidden_units, activation="relu")]
        self.actor = tf.keras.layers.Dense(num_actions)
        self.critic = tf.keras.layers.Dense(1)
        self.emulator = emulator
        self.discountFactor = discount_factor
        self.optimizer = optimizer
        self.criticLoss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)
        self.maxStepsPerEpisode = max_steps_per_episode

    def setCommonLayers(self, common_layers: Sequence[tf.keras.layers.Layer]) -> None:
        assert len(common_layers), "common_layers must have at least one layer object"
        self.commonLayers = common_layers

    def call(self, inputs: tf.Tensor, **kwargs) -> Tuple[tf.Tensor, tf.Tensor]:
        x = self.commonLayers[0](inputs)
        for layer in self.commonLayers[1:]:
            x = layer(x)
        return self.actor(x), self.critic(x)

    def loss(self, action_probs: tf.Tensor, values: tf.Tensor, returns: tf.Tensor) -> tf.Tensor:
        """ combined actor-critic loss. returns is the target value """
        advantage = returns - values
        action_log_prob = tf.math.log(action_probs)
        actor_loss = -tf.math.reduce_sum(action_log_prob * advantage)
        critic_loss = self.criticLoss(values, returns)
        return actor_loss + critic_loss

    def runEpisode(self, initial_state: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        rewards = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

        initial_state_shape = initial_state.shape
        state = initial_state
        for t in tf.range(self.maxStepsPerEpisode):
            state = tf.expand_dims(state, 0)
            action_logits_t, value = self.call(state)
            action = tf.random.categorical(action_logits_t, 1)[0, 0]
            action_probs_t = tf.nn.softmax(action_logits_t)

            values = values.write(t, tf.squeeze(value))
            action_probs = action_probs.write(t, action_probs_t[0, action])

            state, reward, done = self.emulator.tfEnvStep(action)
            rewards = rewards.write(t, reward)

            if tf.cast(done, tf.bool):
                break
        action_probs = action_probs.stack()
        values = values.stack()
        rewards = rewards.stack()
        return action_probs, values, rewards

    def getExpectedReturns(self, rewards: tf.Tensor) -> tf.Tensor:
        """ Expected returns """
        ntime = tf.shape(rewards)[0]
        returns = tf.TensorArray(dtype=tf.float32, size=ntime)

        rewards = tf.cast(rewards[::-1], dtype=tf.float32)
        discounted_sum = tf.constant(0.0)
        discounted_sum_shape = discounted_sum.shape
        for i in tf.range(ntime):
            reward = rewards[i]
            discounted_sum = reward + self.discountFactor * discounted_sum
            discounted_sum.set_shape(discounted_sum_shape)
            returns = returns.write(i, discounted_sum)

        returns = returns.stack()[::-1]

        returns = ((returns - tf.math.reduce_mean(returns)) / (tf.math.reduce_std(returns) + 1E-4))
        return returns

    def train(self, initial_state: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        with tf.GradientTape() as tape:
            action_probs, values, rewards = self.runEpisode(initial_state)
            returns = self.getExpectedReturns(rewards)
            action_probs, values, returns = [tf.expand_dims(x, 1) for x in [action_probs, values, returns]]
            loss = self.loss(action_probs, values, returns)
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        episode_reward = tf.math.reduce_sum(rewards)
        return episode_reward, loss


class AdvantageActorCriticLearner(ReinforcementLearner):
    """
    A2C learner. Needs a value function because it uses the policy being learned by actor.
    Batch A2:
    1. Sample {si, ai} from pi_theta(a|s)
    2. Fit value function V_phi_pi(s) to samples reward sums
    3. Calculate advantage: A_pi(si, ai) = r(si, ai) + gamma*V_phi_pi(s_i+1) - V_phi_pi(si)
    4. grad_theta(J(theta)) = sum_i(grad_theta log(pi_theta(ai|si) * A_pi(si, ai)
    5. theta += alpha * grad_theta J(theta)

    Online A2C:
    1. Take action a ~ pi_theta(a|s) to get (s, a, r, s')
    2. Update V_phi_pi(s) using target r + gamma * V_phi_pi(s')
    3. Calculate advantage: A_pi(si, ai) = r(si, ai) + gamma*V_phi_pi(s_i+1) - V_phi_pi(si)
    4. grad_theta(J(theta)) = sum_i(grad_theta log(pi_theta(ai|si) * A_pi(si, ai)
    5. theta += alpha * grad_theta J(theta)

    Handles the batch A2C version. Online version is a special case of batch version with batch size = 1
    """

    def __init__(self, ac_network):
        """
        Initialize A2C learner
        :param ac_network: Actor-Critic network. Must be an instance of CombinedACNetwork
        """
        assert isinstance(ac_network, CombinedACNetwork)
        self.acNetwork = ac_network

    def fit(self, episodes: list) -> pd.DataFrame:
        assert len(episodes)
        nelements = sum([len(ep) for ep in episodes])
        rewards = np.zeros(nelements, dtype=np.float32)
        losses = np.zeros(nelements, dtype=np.float32)
        assert isinstance(episodes[0], Episode)
        count = 0
        for episode in episodes:
            for initial_sample in episode:
                state, action, reward, next_state = initial_sample
                initial_state = tf.constant(state, dtype=tf.float32)
                episode_reward, loss = self.acNetwork.train(initial_state)
                rewards[count] = episode_reward.numpy()
                losses[count] = loss.numpy()
                count += 1
        return pd.DataFrame({"rewards": rewards, "loss": losses})

    def predict(self, curr_state):
        assert len(curr_state.shape) == 1
        state = tf.constant(curr_state[np.newaxis, :], dtype=tf.float32)
        action_logits_t, value = self.acNetwork.call(state)
        action = tf.math.argmax(action_logits_t, axis=1)
        return action.numpy()[0], value.numpy()[0, 0]

