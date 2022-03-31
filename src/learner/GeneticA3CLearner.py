from __future__ import absolute_import, print_function
from .A3CLearner import A3CLearner, ActorCriticPair
import os
import logging
import threading
import numpy as np
from copy import deepcopy

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
hdlr = logging.FileHandler('gena3c.log')
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr)
logger.setLevel(logging.WARNING)

class GeneticActorCriticPair(ActorCriticPair):
    MUTATION_PROB = 1E-2

    def __init__(self, parent, val_func, policy, thread_freq, episodes, pop_per_thread, ga_steps_per_iter=1):
        self.parent = parent
        self.threadFreq = thread_freq
        self.episodes = episodes
        self.localCount = 0
        self.popPerThread = pop_per_thread
        assert(self.popPerThread % 2 == 0, 'pop_per_thread must be even! Val provided: %d'%pop_per_thread)
        self.policies = self.createPolicies(policy, pop_per_thread)
        self.valFuncs = self.createValueFunctions(val_func, pop_per_thread)
        self.gaStepsPerIter = ga_steps_per_iter

    def createPolicies(self, policy, num_policies):
        policies = [None] * num_policies
        orig_params = policy.getParameters()
        lim = 1E-2
        for i in range(num_policies):
            policies[i] = deepcopy(policy)
            params = deepcopy(orig_params)
            if isinstance(params, list):
                for j in range(len(params)):
                    # random number from uniform dist [-lim, lim)
                    noise = 2*lim*np.random.random(params[j].shape) - lim
                    params[j] = np.add(params[j], noise)
            else:
                assert(isinstance(params, np.ndarray), 'params must be list or numpy array')
                noise = 2 * lim * np.random.random(params.shape) - lim
                params = np.add(params, noise)
            policies[i].copyParameters(params)
        return policies

    def createValueFunctions(self, val_func, num):
        val_funcs = [None]*num
        orig_params = val_func.getParameters()
        lim = 1E-2
        for i in range(num):
            val_funcs[i] = deepcopy(val_func)
            params = deepcopy(orig_params)
            if isinstance(params, list):
                for j in range(len(params)):
                    # random number from uniform dist [-lim, lim)
                    noise = 2 * lim * np.random.random(params[j].shape) - lim
                    params[j] = np.add(params[j], noise)
            else:
                assert (isinstance(params, np.ndarray), 'params must be list or numpy array')
                noise = 2 * lim * np.random.random(params.shape) - lim
                params = np.add(params, noise)
            val_funcs[i].copyParameters(params)
        return val_funcs

    def reproduction(self, weights):
        sums = np.cumsum(weights)
        roulette_wheel = np.divide(sums, sums[sums.shape[0]-1])
        random_draws = np.random.random(self.popPerThread)
        selected_policies = np.searchsorted(roulette_wheel, random_draws, side='left')

        return selected_policies

    def crossoverParameters(self, param1, param2):
        for i in range(len(param1)):
            if isinstance(param1[i], np.ndarray):
                self.crossoverParameters(param1[i], param2[i])

            if isinstance(param1[i], np.float32):
                bits = 63
            else:
                bits = 31
            draw = int(bits*np.random.random())
            p11 = (param1[i] >> draw) << draw
            p21 = (param2[i] >> draw) << draw
            p12 = param1[i] - p11
            p22 = param2[i] - p21
            param1[i] = p11 + p22
            param2[i] = p21 + p12

    def crossover(self, selected):
        # randomly select mating pairs
        mating_pairs = np.arange(self.popPerThread)
        np.random.shuffle(mating_pairs)
        new_policy_params = [None] * self.popPerThread
        new_vfunc_params = [None] * self.popPerThread

        for i in range(0, self.popPerThread, 2):
            policy1 = self.policies[selected[mating_pairs[i]]]
            policy2 = self.policies[selected[mating_pairs[i+1]]]
            param1 = deepcopy(policy1.getParameters())
            param2 = deepcopy(policy2.getParameters())
            self.crossoverParameters(param1, param2)
            new_policy_params[i] = param1
            new_policy_params[i + 1] = param2

            valfunc1 = self.valFuncs[selected[mating_pairs[i]]]
            valfunc2 = self.valFuncs[selected[mating_pairs[i+1]]]
            param1 = deepcopy(valfunc1.getParameters())
            param2 = deepcopy(valfunc2.getParameters())
            self.crossoverParameters(param1, param2)
            new_vfunc_params[i] = param1
            new_vfunc_params[i + 1] = param2

        return new_policy_params, new_vfunc_params

    def mutateParam(self, param):
        if isinstance(param, list):
            for p in param:
                self.mutateParam(p)
            return
        assert(isinstance(param, np.ndarray))
        for i,v in enumerate(param):
            draw = int(31*np.random.random())
            bit = v & (1 << draw)
            if bit:
                # set the bit off
                param[i] = param[i] - bit
            else:
                param[i] = param[i] + (1 << draw)

    def mutation(self, policy_params, vfunc_params):
        self.mutateParam(policy_params)
        self.mutateParam(vfunc_params)

    def geneticEvolutionStep(self, weights):
        selected = self.reproduction(weights)
        new_pol_params, new_vf_params = self.crossover(selected)
        if np.random.random() < self.MUTATION_PROB:
            self.mutation(new_pol_params, new_vf_params)

        for i in range(self.popPerThread):
            self.policies[i].copyParameters(new_pol_params[i])
            self.valFuncs[i].copyParameters(new_vf_params[i])

    def applyCorrections(self, clear=True):
        for policy in self.policies:
            policy.applyCorrection(clear=clear)

        for valfunc in self.valFuncs:
            valfunc.applyCorrection(clear=clear)

    def copyParameters(self, params):
        pol_params, vf_params = params
        for param, policy in zip(pol_params, self.policies):
            policy.copyParameters(param)

        for param, vf in zip(vf_params, self.valFuncs):
            vf.copyParameters(param)

    def run(self):
        logger.info('pid = %d, thread = %s', os.getpid(), threading.current_thread().name)
        policy_weight_arr = np.zeros(self.popPerThread, dtype=np.float32)
        valfunc_weight_arr = np.zeros(self.popPerThread, dtype=np.float32)

        # genetic algorithm evolution step
        for episode in self.episodes:
            for ep_sample in episode:
                ct = 0
                for policy, val_func in zip(self.policies, self.valFuncs):
                    policy_weight_arr[ct] = self.unrollPolicy(policy, val_func, ep_sample)
                    valfunc_weight_arr[ct] = 1.0/np.abs(val_func.value(ep_sample.state) - policy_weight_arr[ct])
                    ct += 1
                policy_weight_arr = np.divide(policy_weight_arr, sum(policy_weight_arr))
                valfunc_weight_arr = np.divide(valfunc_weight_arr, sum(valfunc_weight_arr))
                combined_weight_arr = np.sum(policy_weight_arr, valfunc_weight_arr)
                self.geneticEvolutionStep(combined_weight_arr)

        # gradient descent (critic), ascent (actor) step
        avg_disc_rewards = np.zeros(self.popPerThread, dtype=np.float32)
        avg_vfunc_perf = np.zeros(self.popPerThread, dtype=np.float32)
        for episode in self.episodes:
            for ep_i, ep_sample in enumerate(episode):
                self.localCount += 1
                j = 0
                for policy, val_func in zip(self.policies, self.valFuncs):
                    disc_reward = self.unrollPolicy(policy, val_func, ep_sample)
                    val_func.learnFromTarget(disc_reward, ep_sample, self.parent.alpha, apply_correction=False)
                    # calculate advantage
                    advantage = disc_reward - val_func.value(ep_sample.state)
                    policy.policyGradientStep(advantage, ep_sample.state, ep_sample.action,
                                              self.parent.alpha, apply_correction=False)
                    disc_reward = self.unrollPolicy(policy, val_func, ep_sample)
                    avg_disc_rewards[j] += disc_reward
                    avg_vfunc_perf[j] += 1.0/abs(disc_reward - val_func.value(ep_sample.state))
                    j += 1

                if self.localCount % self.threadFreq == 0:
                    # apply corrections to local network parameters
                    # apply gradient ascent/descent corrections to local policy and value functions
                    self.applyCorrections(clear=True)

                    # Offer parameters to parent network, parent selects according to its strategy (best, average etc)
                    # copy the parent network parameters to local network parameters
                    # parent will copy the params if they are better than existing ones (sorted)
                    rewards = np.sum(avg_disc_rewards, avg_vfunc_perf) / self.threadFreq
                    begin = max(0, ep_i-self.threadFreq)
                    self.parent.offerParameters(self.policies, self.valFuncs, rewards, episode[begin:ep_i+1])
                    avg_disc_rewards[:] = 0.0
                    avg_vfunc_perf[:] = 0.0
                    # copy parameters from the parent
                    self.copyParameters(self.parent.getParameters())


class GeneticA3CLearner(A3CLearner):
    '''
    Genetic A3C learner
    '''
    def __init__(self, val_func, policy, alpha, gamma,
                 reward_calculator, state_emulator, epsilon=0.0,
                 update_thread_freq=10, tdn=0, nthreads=10,
                 pop_per_thread=10):
        super(self.__class__, self).__init__(val_func, policy, alpha, gamma,
                 reward_calculator, state_emulator, epsilon=epsilon,
                 update_thread_freq=update_thread_freq, tdn=tdn, nthreads=nthreads)
        self.popPerThread = pop_per_thread
        self.avgDiscRewards = np.zeros(self.popPerThread, dtype=np.float32)
        self.avgVfuncPerf = np.zeros(self.popPerThread, dtype=np.float32)
        self.policies = [None] * self.popPerThread
        self.valFuncs = [None] * self.popPerThread

    def unrollPolicy(self, policy, val_func, episode_sample):
        sum_rwd = 0.0
        fac = 1.0
        state = episode_sample.state
        next_state = episode_sample.next_state
        for j in range(self.tdn+1):
            if state is None:
                return sum_rwd
            action = policy.getNextAction(state)
            next_state = self.stateEmulator.nextState(state, action, next_state)
            reward = self.rewardCalculator.calculate(state, action, next_state)
            sum_rwd += fac*reward
            fac *= self.gamma
            state = next_state

        if state is None:
            return sum_rwd
        sum_rwd += fac*val_func.value(state)
        return sum_rwd

    def createActorCriticPair(self, vfunc, policy, thread_freq, episodes):
        return GeneticActorCriticPair(self, vfunc, policy, thread_freq, episodes, self.popPerThread)

    def offerParameters(self, policies, val_funcs, rewards, episode_samples):
        with self.lock:
            if self.policies[0] is None:
                self.policies[:] = policies
                self.valFuncs[:] = val_funcs
                return
            self.avgDiscRewards[:] = 0.0
            self.avgVfuncPerf[:] = 0.0
            for i in range(self.popPerThread):
                for ep_sample in episode_samples:
                    disc_reward = self.unrollPolicy(self.policies[i], self.valFuncs[i], ep_sample)
                    self.avgDiscRewards[i] += disc_reward
                    self.avgVfuncPerf[i] += 1.0 / abs(disc_reward - self.valFuncs[i].value(ep_sample.state))

            existing_rewards = np.sum(self.avgDiscRewards, self.avgVfuncPerf) / len(episode_samples)
            erew_arr = [(r,i) for i,r in enumerate(existing_rewards)]
            rew_arr = [(r,i) for i,r in enumerate(rewards)]
            # sort the two arr
            erew_arr.sort(key=lambda x: x[0], reverse=True)
            rew_arr.sort(key=lambda x: x[0], reverse=True)
            b1 = 0
            b2 = 0
            indx = 0
            while (b1 != len(erew_arr)) and (b2 != len(rew_arr)):
                if erew_arr[b1][0] < rew_arr[b2][0]:
                    self.policies[indx] = policies[rew_arr[b2][1]]
                    self.valFuncs[indx] = val_funcs[rew_arr[b2][1]]
                    b2 += 1
                else:
                    self.policies[indx] = policies[erew_arr[b1][1]]
                    self.valFuncs[indx] = val_funcs[erew_arr[b1][1]]
                    b1 += 1
                indx += 1
            if b1 != len(erew_arr):
                self.policies[indx:] = [policies[erew_arr[b][1]] for b in range(b1, len(erew_arr))]
                self.valFuncs[indx:] = [val_funcs[erew_arr[b][1]] for b in range(b1, len(erew_arr))]
            elif b2 != len(rew_arr):
                self.policies[indx:] = [policies[rew_arr[b][1]] for b in range(b1, len(rew_arr))]
                self.valFuncs[indx:] = [val_funcs[rew_arr[b][1]] for b in range(b1, len(rew_arr))]

    def getParameters(self):
        with self.lock:
            pol_params = [p.getParameters() for p in self.policies]
            params = [vf.getParameters() for vf in self.valFuncs]
            return pol_params, params
