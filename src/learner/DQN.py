import numpy as np
import pandas as pd

import src.lib.Emulator as Emulator
import src.lib.QFunction as QFunc
import src.lib.ReplayBuffer as RB
from src.lib.Episode import Episode
from src.lib.ReinforcementLearner import ReinforcementLearner
from src.lib.Sample import Sample


class DQN(ReinforcementLearner):
    """ Deep Q Network learner """
    def __init__(self, q_function, emulator, state_size, num_actions, discount_factor=0.9, minibatch_size=10,
                 replay_buffer=None, epochs_training=10):
        """
        Initialize the DQN learner. Operates on discrete action spaces.
        :param q_function: Action value function. Object of type QFunction. Could be a TensorFlow model
        wrapped in QNeuralNet class
        :param emulator: An object of type StateAndRewardEmulator gives the reward and next state from
        (state, action) input
        :param state_size: Dimension of state space
        :param num_actions: Number of actions
        :param discount_factor:
        :param minibatch_size: Size of minibatch for stochastic gradient descent
        :param replay_buffer: An object of type ReplayBuffer
        :param epochs_training: Number of epochs used in training
        """
        assert isinstance(q_function, QFunc.QFunction)
        assert isinstance(emulator, Emulator.StateAndRewardEmulator)
        self.qFunc = q_function
        self.minibatchSize = minibatch_size
        self.emulator = emulator
        self.numActions = num_actions
        self.stateSize = state_size
        self.discountFactor = discount_factor
        self.epochs = epochs_training
        self.trainingTargets = np.zeros((self.minibatchSize, num_actions), dtype=np.float32)
        self.trainingInputs = np.zeros((self.minibatchSize, state_size), dtype=np.float32)
        if replay_buffer is None:
            self.replayBuffer = RB.SimpleReplayBuffer(capacity=2*minibatch_size)
        else:
            assert isinstance(replay_buffer, RB.ReplayBuffer)
            self.replayBuffer = replay_buffer

    def _improveQFunc(self, df):
        """ Improve Q function using TD(0) learning
            :param df: dataframe to hold or append history of fitting
            :return dataframe containing the appended history
         """
        if len(self.replayBuffer) < self.minibatchSize:
            return
        minibatch = self.replayBuffer.sampleMinibatch(self.minibatchSize)
        targets = self.trainingTargets
        inputs = self.trainingInputs
        for i, sample in enumerate(minibatch):
            state, action, reward, next_state = sample
            qval = reward
            if next_state is not None:
                qval = reward + self.discountFactor * np.max(self.qFunc.predict(next_state[np.newaxis, :])[0])
            targets[i, :] = self.qFunc.predict(state[np.newaxis, :])[0]
            targets[i, action] = qval
            inputs[i, :] = state
        history = self.qFunc.fit(inputs, targets, epochs=self.epochs)
        if df is None:
            df = pd.DataFrame(history.history)
        else:
            df = df.append(history.history, ignore_index=True)
        return df

    def fit(self, episodes):
        """
        Train the DQN learner using episodes
        :param episodes: A list (or iterable) containing episodes. Each episode of type Episode
        :return: Dataframe containing the history of training (includes metrics)
        """
        df = None
        assert isinstance(episodes[0], Episode)

        insert_count = 0
        time_step = 0
        for episode in episodes:
            for initial_sample in episode:
                state, action, reward, next_state = initial_sample
                done = False
                while not done:
                    action, qval = self.predict(state)
                    next_state, reward, done, info = self.emulator.step(state, action)
                    if done:
                        reward = -reward
                        next_state = None
                    sample = Sample(state, action, reward, next_state)
                    self.replayBuffer.addExperience(sample)
                    insert_count += 1
                    time_step += 1
                    state = next_state
                    if insert_count == self.minibatchSize:
                        df = self._improveQFunc(df)
                        insert_count = 0

                    # the step below does nothing for DQN, for DDQN it will update target network weights periodically
                    self._postTimeStepProcessing(time_step)

        if insert_count:
            df = self._improveQFunc(df)
        return df

    def predict(self, curr_state):
        """
        Get the next optimum action and total reward at a state using the trained value function
        :param curr_state: Current state
        :return: optimum action, total reward
        """
        assert len(curr_state.shape) == 1
        curr_state = curr_state[np.newaxis, :]
        qvals = self.qFunc.predict(curr_state)
        index = np.argmax(qvals[0])
        return index, qvals[0, index]

    def _postTimeStepProcessing(self, time_step):
        pass
