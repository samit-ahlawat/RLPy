from __future__ import absolute_import, print_function

import logging
import os
import threading
from copy import deepcopy

import numpy as np

import src.lib.Emulator as Emulator
import src.lib.Policy as Policy
import src.lib.ValFunction as VFunc
from src.lib.ReinforcementLearner import ReinforcementLearner

# logging
FORMAT = '%(asctime)s %(relativeCreated)6d %(threadName)s %(name)s %(levelname)s %(message)s'
logging.basicConfig(format=FORMAT, level=logging.DEBUG)


class ActorCriticPair(threading.Thread):
    """ Actor critic pair. Runs in a separate thread. """
    def __init__(self, parent, val_func, policy, thread_freq, episodes, epochs, log_level):
        assert len(episodes)
        super(self.__class__, self).__init__()
        self.parent = parent
        self.valFunc = val_func
        self.policy = policy
        self.threadFreq = thread_freq
        self.episodes = episodes
        self.epochs = epochs
        self.logger = logging.getLogger('A3CLearner.ActorCriticPair')
        self.logger.setLevel(log_level)

    def unrollPolicy(self, state):
        sum_rwd = 0.0
        fac = 1.0
        gamma = self.parent.gamma
        for j in range(self.parent.tdn+1):
            if state is None:
                return sum_rwd
            action = self.policy.getNextAction(state)
            reward, next_state = self.parent.emulator.getRewardAndNextState(state, action)
            sum_rwd += fac*reward
            fac *= gamma
            state = next_state

        if state is None:
            return sum_rwd
        sum_rwd += fac*self.valFunc.value(state)
        return sum_rwd

    def calculateAdvantage(self, state, reward, next_state):
        """
        Calculate advantage as A(r,s) = reward + gamma * V(next_state) - V(state)
        :param state:
        :param reward:
        :param next_state:
        :return: advantage
        """
        if next_state is None:
            next_value = reward
        else:
            state_vals = np.array(next_state.values(), dtype=np.float32)
            val = self.valFunc.predict(state_vals[np.newaxis, :])
            next_value = reward + self.parent.gamma * val[0]
        state_vals = np.array(state.values(), dtype=np.float32)
        val = self.valFunc.predict(state_vals[np.newaxis, :])
        value = val[0]
        return value - next_value

    def run(self):
        self.logger.info('Thread started: pid = %d, thread = %s', os.getpid(), threading.current_thread().name)
        local_count = 0
        input_size = len(self.episodes[0][0].state)
        training_targets = np.zeros(self.threadFreq, dtype=np.float32)
        training_inputs = np.zeros((self.threadFreq, input_size), dtype=np.float32)

        for episode in self.episodes:
            for ep_sample in episode:
                disc_reward = self.unrollPolicy(ep_sample.state)
                training_targets[local_count] = disc_reward
                training_inputs[local_count, :] = ep_sample.state.values()
                local_count += 1

                if local_count % self.threadFreq == 0:
                    valfunc_params = deepcopy(self.valFunc.getParameters())
                    policy_params = deepcopy(self.policy.getParameters())

                    # calculate corrections corresponding to threadFreq observations
                    self.valFunc.fit(training_inputs, training_targets, epochs=self.epochs)
                    self.policy.fit(training_inputs, epochs=self.epochs)
                    new_valfunc_params = self.valFunc.getParameters()
                    new_policy_params = self.policy.getParameters()
                    valfunc_corrections = self.valFunc.getParamCorrections(valfunc_params, new_valfunc_params)
                    policy_corrections = self.policy.getParamCorrections(policy_params, new_policy_params)

                    # send corrections to parent network
                    self.parent.applyCorrections(valfunc_corrections, policy_corrections)
                    # self.logger.info('Thread applied corrections to parent params')

                    # copy parent network parameters to local network parameters
                    self.valFunc.setParameters(self.parent.getValFuncParameters())
                    self.policy.setParameters(self.parent.getPolicyParameters())

                    # reset the count to 0
                    local_count = 0


class A3CLearner(ReinforcementLearner):
    """
    A3C learner: Similar in principle to A2C learner. Differs in following respects:
    Does not use replay buffer
    Uses multiple threads executing actor-critic pair
    """

    def __init__(self, value_function, policy, emulator, discount_factor=0.9,
                 epochs_training=10, tdn=0, update_thread_freq=10, nthreads=10,
                 log_level=logging.DEBUG):
        """
        Initialize A3C learner
        :param value_function: State value function. Object of type QFunction. Could be a TensorFlow model wrapped in QNeuralNet class
        :param policy: Policy to be learned. Could ne a neural network wrapped in a PolicyNeuralNet class
        :param emulator: State and rewards emulator. Given a state and an action, gives the reward and the next state
        :param discount_factor: Discount factor to weigh future rewards
        :param epochs_training: Number of epochs used in training
        :param tdn: Use a TD(n) target, with n steps of policy unrolling. Higher the n, less is it's bias (but with higher variance)
        :param update_thread_freq: Send the weight corrections from child thread to parent after processing update_thread_freq samples
        :param nthreads: Number of actor-critic pairs
        :param log_level: Logging level
        """
        assert isinstance(value_function, VFunc.ValFunction)
        assert isinstance(emulator, Emulator.StateAndRewardEmulator)
        assert isinstance(policy, Policy.Policy)
        assert nthreads > 1
        assert tdn >= 0
        self.valFunc = value_function
        self.policy = policy
        self.gamma = discount_factor
        self.emulator = emulator
        self.epochs = epochs_training
        self.tdn = tdn
        self.nThreads = nthreads
        self.threadFreq = update_thread_freq
        self.lock = threading.Lock()
        self.threads = [None] * self.nThreads
        self.logger = logging.getLogger('A3CLearner')
        self.logLevel = log_level
        self.logger.setLevel(log_level)

    def applyCorrections(self, vfunc_corr, policy_corr):
        """
        Apply corrections to value function and policy. Takes care of concurrency.
        :param vfunc_corr:
        :param policy_corr:
        """
        with self.lock:
            self.valFunc.applyParamCorrections(vfunc_corr)
            self.policy.applyParamCorrections(policy_corr)

    def getValFuncParameters(self):
        """
        Get value function trainable parameters. Takes care of concurrency.
        :return: trainable parameters of value function
        """
        with self.lock:
            return self.valFunc.getParameters()

    def getPolicyParameters(self):
        """
        Get policy trainable parameters. Takes care of concurrency.
        :return: trainable parameters of policy
        """
        with self.lock:
            return self.policy.getParameters()

    def createActorCriticPair(self, thread_freq, episodes):
        """
        Create a actor critic pair using a copy of the value function and policy
        :param thread_freq: Number of steps after which child should send value function
        and policy param corrections to the parent
        :param episodes: list of episodes to learn from
        :return: A thread object that can run an actor-critic pair
        """
        vfunc = deepcopy(self.valFunc)
        policy = deepcopy(self.policy)
        return ActorCriticPair(self, vfunc, policy, thread_freq, episodes, self.epochs, self.logLevel)

    def fit(self, episodes):
        """
        Fit the value function and policy to the provided episodes.
        :param episodes: A list of episodes
        """
        # rng = np.random.default_rng()
        permuted_episodes = np.random.permutation(len(episodes))
        # divide the episodes equally among the child jobs
        div = int(len(episodes)/self.nThreads)
        limits = [i*div for i in range(self.nThreads)]
        limits.append(len(episodes))
        for tnum in range(self.nThreads):
            episodes_slice = permuted_episodes[limits[tnum]:limits[tnum+1]]
            episodes_list = [episodes[i] for i in episodes_slice]
            self.threads[tnum] = self.createActorCriticPair(self.threadFreq, episodes_list)
            self.threads[tnum].start()

        self.logger.info("Launched all actor-critic threads")
        for th in self.threads:
            th.join()
        self.logger.info("Training complete")

    def predict(self, curr_state):
        """
        Predict the next optimal action and total discounted reward starting from curr_state
        Uses the trained action and value functions
        :param curr_state: np.ndarray containing a set of current states. Dimension: #batches X #features in a state
        :return Next optimal action and total discounted reward
        """
        action = self.policy.getNextAction(curr_state)
        # value = self.valFunc.predict(curr_state)
        return action