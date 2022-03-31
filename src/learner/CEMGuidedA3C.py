from __future__ import absolute_import, print_function

import functools
import logging
import os
import threading
from copy import deepcopy

import numpy as np

import src.lib.Emulator as Emulator
import src.lib.Policy as Policy
import src.lib.ValFunction as VFunc
from src.lib.Emulator import StateAndRewardEmulator
from src.lib.ReinforcementLearner import ReinforcementLearner
from src.lib.State import State

# logging
FORMAT = '%(asctime)s %(relativeCreated)6d %(threadName)s %(name)s %(levelname)s %(message)s'
logging.basicConfig(format=FORMAT, level=logging.DEBUG)


class CrossEntropyPolicy(Policy.Policy):
    """
        Cross Entropy Guided Policy
        The underlying policy is a neural network (TensorFlow).
        Trains the network parameters using posterior bayesian update rule.
    """

    def randomAction(self, state):
        return self.policy.randomAction(state)

    def __init__(self, val_func, policy, emulator, niters, params_shape, sample_size, percentile,
                 disc_factor=0.99, weight=0.9):
        """

        :param niters:
        :param params_shape:
        """
        assert isinstance(policy, Policy.Policy)
        assert isinstance(val_func, VFunc.ValFunction)
        assert isinstance(emulator, StateAndRewardEmulator)
        self.valFunc = val_func
        self.policy = policy
        self.nIters = niters
        self.emulator = emulator
        assert isinstance(params_shape, (list, tuple))
        self.paramsShape = params_shape
        self.meanVec = [np.zeros(sh) for sh in params_shape]
        self.stdVec = [np.ones(sh) for sh in params_shape]
        self.paramCountList = [functools.reduce(lambda x, y=1: x*y, sh) for sh in params_shape]
        self.paramCount = np.sum(self.paramCountList)
        self.sampleSize = sample_size
        self.percentile = percentile
        self.level = int(sample_size * percentile)
        self.discFactor = disc_factor
        self.weight = weight

    def posteriorProb(self, param, mean_vec, std_vec):
        mean_vec[...] = self.weight * mean_vec + (1 - self.weight) * param

    def crossEntropyIterations(self, states):
        params = self.policy.getParameters()
        if isinstance(params, np.ndarray):
            params = [params]
        assert isinstance(params, (list, tuple))
        assert len(params) == len(self.paramsShape)
        noisy_params = [[]] * self.sampleSize
        for i0 in range(self.sampleSize):
            noisy_params[i0] = [[]] * len(params)
            for i in range(len(params)):
                noisy_params[i0][i] = [np.ndarray(params[i][0].shape, dtype=np.float32), params[i][1]]

        for i in range(len(params)):
            # assign meanVec, stdVec using posterior density
            self.posteriorProb(params[i][0], self.meanVec[i], self.stdVec[i])

        for i in range(self.nIters):
            self.crossEntropyIter(states, noisy_params)

        layer_params = [[mv, params[i][1]] for i,mv in enumerate(self.meanVec)]
        self.policy.setParameters(layer_params)

    def crossEntropyIter(self, states, noisy_params):
        dtype_rewards = [('rewards', np.float32), ('order', np.int)]
        rewards_arr = np.ndarray(self.sampleSize, dtype=dtype_rewards)  # structured array
        for j in range(self.sampleSize):
            rewards_arr[j]['rewards'] = 0.0
            rewards_arr[j]['order'] = j
            eps = np.random.standard_normal(self.paramCount)
            begin = 0
            # TODO: what is inside self.paramsShape, self.paramCountList
            for i in range(len(self.paramsShape)):
                end = begin + self.paramCountList[i]
                eps_i = eps[begin:end].reshape(self.paramsShape[i])
                noisy_params[j][i][0] = self.meanVec[i] + np.multiply(eps_i, self.stdVec[i])
                begin = end
            # perturb the mean, generate actions
            self.policy.setParameters(noisy_params[j])
            actions = self.policy.getNextAction(states)
            for st, act in zip(states, actions):
                reward, next_state = self.emulator.getRewardAndNextState(st, act)
                if isinstance(next_state, State):
                    next_state = np.array(next_state.values(), dtype=np.float32)[np.newaxis, :]
                vals = self.valFunc.predict(next_state)
                rewards_arr[j]['rewards'] += reward + self.discFactor*vals[0]
            rewards_arr[j]['rewards'] /= float(len(states))

        rewards_arr = np.sort(rewards_arr, order='rewards')
        indices = [rewards_arr[i]['order'] for i in range(self.level, self.sampleSize)]
        # mean of perm_params[indices, :]
        best_params = [noisy_params[ind] for ind in indices]
        for i in range(len(self.paramsShape)):
            arrays = [best_params[j][i][0] for j in range(len(best_params))]
            arrsq = [np.multiply(ar, ar) for ar in arrays]
            self.meanVec[i] = sum(arrays)/len(arrays)
            self.stdVec[i] = sum(arrsq)/len(arrays) - np.multiply(self.meanVec[i], self.meanVec[i])

    def fit(self, training_inputs, *args, **kwargs):
        """
        Fit the data. Only uses states passed in training_inputs, in conjunction with policy, reward emulator and
        value function
        :param training_inputs:
        :param args:
        :param kwargs:
        """
        self.crossEntropyIterations(training_inputs)

    def predict(self, input, *args, **kwargs):
        return self.policy.predict(input, *args, **kwargs)

    def setParameters(self, params):
        mean_vec, std_vec = params
        for i in range(len(self.meanVec)):
            self.meanVec[i][...] = mean_vec[i]

        for i in range(len(self.stdVec)):
            self.stdVec[i][...] = std_vec[i]

    def getParameters(self):
        return self.meanVec, self.stdVec

    def getParamCorrections(self, params, new_params):
        pass

    def applyParamCorrections(self, corrections):
        pass

    def getNextAction(self, state):
        return self.policy.getNextAction(state)

# cross entropy guided policy and value function guided policy in actor
# QGP guided value function in critic


class ActorCriticPair(threading.Thread):
    """ Actor critic pair. Runs in a separate thread. """
    def __init__(self, parent, val_func, policy, thread_freq, episodes, epochs, log_level, cem_iters=10,
                 cem_sample_size=20, cem_percentile=0.9):
        assert len(episodes)
        super(self.__class__, self).__init__()
        self.parent = parent
        self.valFunc = val_func
        self.policy = policy
        params = policy.getParameters()
        params_shape = [param[0].shape for param in params]
        self.cemPolicy = CrossEntropyPolicy(val_func, policy, parent.emulator, cem_iters,
                                            params_shape, cem_sample_size, cem_percentile, weight=0.3)
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
                    valfunc_params = self.valFunc.getParameters()
                    policy_params = self.policy.getParameters()

                    # calculate corrections corresponding to threadFreq observations
                    self.valFunc.fit(training_inputs, training_targets, epochs=self.epochs)
                    self.policy.fit(training_inputs, epochs=self.epochs)
                    self.cemPolicy.fit(training_inputs, None)
                    new_valfunc_params = self.valFunc.getParameters()
                    new_policy_params = self.policy.getParameters()
                    valfunc_corrections = self.valFunc.getParamCorrections(valfunc_params, new_valfunc_params)
                    policy_corrections = self.policy.getParamCorrections(policy_params, new_policy_params)

                    # send corrections to parent network
                    self.parent.applyCorrections(valfunc_corrections, policy_corrections)
                    self.logger.info('Thread applied corrections to parent params')

                    # copy parent network parameters to local network parameters
                    self.valFunc.setParameters(self.parent.getValFuncParameters())
                    self.policy.setParameters(self.parent.getPolicyParameters())

                    # reset the count to 0
                    local_count = 0


class CEMA3Learner(ReinforcementLearner):
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
        Predict the next optimal action starting from curr_state
        Uses the trained action functions
        :param curr_state: np.ndarray containing a set of current states. Dimension: #batches X #features in a state
        :return Next optimal action
        """
        action = self.policy.getNextAction(curr_state)
        # value = self.valFunc.predict(curr_state)
        return action