import gym
import os
import argparse
import numpy as np
from fourrooms import Fourrooms
from scipy.misc import logsumexp
from scipy.special import expit

class Tabular:
    def __init__(self, nstates):
        self.nstates = nstates

    def __call__(self, state):
        return np.array([state,])

    def __len__(self):
        return self.nstates

class EgreedyPolicy:
    def __init__(self, rng, nfeatures, nactions, epsilon):
        self.rng = rng
        self.epsilon = epsilon
        self.nactions = nactions
        self.weights = np.zeros((nfeatures, nactions)) # weight initialization

    def value(self, phi, action=None):
        if action is None:
            return np.sum(self.weights[phi, :], axis=0)
        return np.sum(self.weights[phi, action], axis=0)


    def sample(self, phi):
        if self.rng.uniform() < self.epsilon:
            return int(self.rng.randint(self.weights.shape[1]))
        return int(np.argmax(self.value(phi)))


class UniformPolicyOverOptions:
    def __init__(self, rng, noptions):
        self.rng = rng
        self.noptions = noptions

    def sample(self, phi):
        return int(self.rng.choice(self.noptions, p=self.pmf(phi)))

    def pmf(self, phi):
        option_distribution = np.ones((self.noptions))*(1/self.noptions)
        return option_distribution


class RoomSpecificPolicyOverOptions:
    def __init__(self, noptions):
        self.noptions = noptions
        self.room1 = list(range(5)) + list(range(10, 15)) + list(range(20, 26)) + list(range(31, 36)) + list(range(41, 46)) + [51]
        self.room2 = list(range(5, 10)) + list(range(15, 20)) + list(range(26, 31)) + list(range(36, 41)) + list(range(46, 51)) + list(range(52, 57))
        self.room3 = list(range(68, 73)) + list(range(78, 83)) + list(range(89, 94)) + list(range(99, 104)) + [62]
        self.room4 = list(range(57, 62)) + list(range(63, 68)) + list(range(73, 78)) + list(range(83, 89)) + list(range(94, 99))

    def sample(self, phi):
        if (phi in self.room1):
            return 0
        elif(phi in self.room2):
            return 1
        elif (phi in self.room3):
            return 2
        elif (phi in self.room4):
            return 3

    def pmf(self, phi):
        option = self.sample(phi)
        option_distribution = np.ones((self.noptions))*0.033
        option_distribution[option] = 0.9
        return option_distribution


class SoftmaxPolicy:
    def __init__(self, rng, nfeatures, nactions, temp=1.):
        self.rng = rng
        self.weights = np.zeros((nfeatures, nactions))
        self.nactions = nactions
        self.temp = temp

    def value(self, phi, action=None):
        if action is None:
            return np.sum(self.weights[phi, :], axis=0)
        return np.sum(self.weights[phi, action], axis=0)

    def pmf(self, phi):
        v = self.value(phi)/self.temp
        return np.exp(v - logsumexp(v))

    def sample(self, phi):
        return int(self.rng.choice(self.nactions, p=self.pmf(phi)))



class SigmoidTermination:
    def __init__(self, rng, nfeatures):
        self.rng = rng
        self.weights = np.zeros((nfeatures,))


    def pmf(self, phi):
        return expit(np.sum(self.weights[phi]))

    def sample(self, phi):
        return int(self.rng.uniform() < self.pmf(phi))

    def grad(self, phi):
        terminate = self.pmf(phi)
        return terminate*(1. - terminate), phi



class IntraOptionQLearning:
    def __init__(self, discount, lr, terminations, weights, policy_over_options):
        self.lr = lr
        self.discount = discount
        self.terminations = terminations
        self.weights = weights
        self.policy_over_options = policy_over_options

    def start(self, phi, option=None):
        self.last_phi = phi
        self.last_option = option
        self.last_value = self.value(phi, option)

    def value(self, phi, option=None):                          # Q(s,w)
        if option is None:
            return np.sum(self.weights[phi, :], axis=0)
        return np.sum(self.weights[phi, option], axis=0)

    def vvalue(self, phi):                                       # V(s)
        values = self.value(phi)                                 # Q(s,w)
        pi = self.policy_over_options.pmf(phi)                   # pi_omega(w|s)
        vvalue = sum([x * y for x, y in zip(values, pi)])
        return vvalue

    def advantage(self, phi, option=None):
        values = self.value(phi)
        advantages = values - np.max(values)
        if option is None:
            return advantages
        return advantages[option]


    def update(self, phi, option, reward, done):
        # One-step update target
        update_target = reward
        if not done:
            current_values = self.value(phi)
            termination = self.terminations[self.last_option].pmf(phi)
            update_target += self.discount*((1. - termination)*current_values[self.last_option] + termination*np.max(current_values))

        # Dense gradient update step
        tderror = update_target - self.last_value
        self.weights[self.last_phi, self.last_option] += self.lr*tderror

        if not done:
            self.last_value = current_values[option]

        self.last_option = option
        self.last_phi = phi

        return update_target


class IntraOptionActionQLearning:
    def __init__(self, discount, lr, terminations, weights, qbigomega):
        self.lr = lr
        self.discount = discount
        self.terminations = terminations
        self.weights = weights
        self.qbigomega = qbigomega

    def value(self, phi, option, action):
        return np.sum(self.weights[phi, option, action], axis=0)

    def start(self, phi, option, action):
        self.last_phi = phi
        self.last_option = option
        self.last_action = action

    def update(self, phi, option, action, reward, done):
        # One-step update target
        update_target = reward
        if not done:
            current_values = self.qbigomega.value(phi)
            termination = self.terminations[self.last_option].pmf(phi)
            update_target += self.discount*((1. - termination)*current_values[self.last_option] + termination*np.max(current_values))

        # Update values upon arrival if desired
        tderror = update_target - self.value(self.last_phi, self.last_option, self.last_action)
        self.weights[self.last_phi, self.last_option, self.last_action] += self.lr*tderror

        self.last_phi = phi
        self.last_option = option
        self.last_action = action


class TerminationGradient:
    def __init__(self, terminations, critic, lr):
        self.terminations = terminations
        self.critic = critic
        self.lr = lr

    def update(self, phi, option):
        magnitude, direction = self.terminations[option].grad(phi)
        self.terminations[option].weights[direction] -= \
                self.lr*magnitude*(self.critic.advantage(phi, option))


class IntraOptionGradient:
    def __init__(self, option_policies, lr):
        self.lr = lr
        self.option_policies = option_policies

    def update(self, phi, option, action, critic):
        actions_pmf = self.option_policies[option].pmf(phi)
        self.option_policies[option].weights[phi, :] -= self.lr*critic*actions_pmf
        self.option_policies[option].weights[phi, action] += self.lr*critic


class OneStepTermination:
    def sample(self, phi):
        return 1

    def pmf(self, phi):
        return 1.


class FixedActionPolicies:
    def __init__(self, action, nactions):
        self.action = action
        self.probs = np.eye(nactions)[action]

    def sample(self, phi):
        return self.action

    def pmf(self, phi):
        return self.probs


def save_params(args, dir_name):
    f = os.path.join(dir_name, "Params.txt")
    with open(f, "w") as f_w:
        for param, val in sorted(vars(args).items()):
            f_w.write("{0}:{1}\n".format(param, val))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--discount', help='Discount factor', type=float, default=0.99)
    parser.add_argument('--lr_term', help="Termination gradient learning rate", type=float, default=0.25)
    parser.add_argument('--lr_intra', help="Intra-option gradient learning rate", type=float, default=0.25)
    parser.add_argument('--lr_critic', help="Learning rate", type=float, default=0.5)
    parser.add_argument('--epsilon', help="Epsilon-greedy for policy over options", type=float, default=1e-2)
    parser.add_argument('--nepisodes', help="Number of episodes per run", type=int, default=2000)
    parser.add_argument('--nruns', help="Number of runs", type=int, default=1)
    parser.add_argument('--nsteps', help="Maximum number of steps per episode", type=int, default=2000)
    parser.add_argument('--noptions', help='Number of options', type=int, default=4)
    parser.add_argument('--baseline', help="Use the baseline for the intra-option gradient", action='store_true', default=True)
    parser.add_argument('--temperature', help="Temperature parameter for softmax", type=float, default=1e-2)
    parser.add_argument('--primitive', help="Augment with primitive", default=False, action='store_true')
    parser.add_argument('--seed', help="seed value for experiment", type=int, default=10)
    parser.add_argument('--seed_startstate', help="seed value for starting state", type=int, default=10)

    args = parser.parse_args()
    rng = np.random.RandomState(args.seed)
    env = Fourrooms(args.seed_startstate)


    outer_dir = "OptionCritic"
    if not os.path.exists(outer_dir):
        os.makedirs(outer_dir)

    dir_name = "Runs"+str(args.nruns)+"_Epsds"+str(args.nepisodes)+ "_Eps"+str(args.epsilon)+"_NOpt"+str(args.noptions)+"_LRT"+ str(args.lr_term) +\
     "_LRI"+str(args.lr_intra)+"_LRC"+str(args.lr_critic)+"_temp"+str(args.temperature)+"_seed"+str(args.seed)

    dir_name = os.path.join(outer_dir, dir_name)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    save_params(args, dir_name)

    possible_next_goals = [68, 69, 70, 71, 72, 78, 79, 80, 81, 82, 88, 89, 90, 91, 92, 93, 99, 100, 101,102, 103]

    num_states = env.observation_space.n
    num_actions = env.action_space.n

    history = np.zeros((args.nruns, args.nepisodes, 3), dtype=np.float32)

    state_frequency_history = np.zeros((args.nruns, args.nepisodes, env.observation_space.n, args.noptions),dtype=np.int32)

    # for storing the weights of the trained model
    weight_intra_option = np.zeros((args.nruns ,args.nepisodes, num_states, num_actions,  args.noptions), dtype=np.float32)
    weight_policy_over_options = np.zeros((args.nruns ,args.nepisodes, num_states, args.noptions), dtype=np.float32)
    weight_termination = np.zeros((args.nruns ,args.nepisodes, num_states, args.noptions), dtype=np.float32)
    # for storing the weights of Q(s,w)
    weight_option_value = np.zeros((args.nruns, args.nepisodes, num_states, args.noptions), dtype=np.float32)

    # for storing the weights of Q(s,w,a)
    weight_action_value = np.zeros((args.nruns, args.nepisodes, num_states, num_actions, args.noptions), dtype=np.float32)

    for run in range(args.nruns):
        features = Tabular(env.observation_space.n)
        nfeatures, nactions = len(features), env.action_space.n


        # The intra-option policies are linear-softmax functions
        option_policies = [SoftmaxPolicy(rng, nfeatures, nactions, args.temperature) for _ in range(args.noptions)]

        if args.primitive:
            option_policies.extend([FixedActionPolicies(act, nactions) for act in range(nactions)])

        # The termination function are linear-sigmoid functions
        option_terminations = [SigmoidTermination(rng, nfeatures) for _ in range(args.noptions)]
        if args.primitive:
            option_terminations.extend([OneStepTermination() for _ in range(nactions)])

        # E-greedy policy over options
        # policy_over_options = EgreedyPolicy(rng, nfeatures, args.noptions, args.epsilon)
        # policy_over_options = SoftmaxPolicy(rng, nfeatures, args.noptions, args.temperature)
        policy_over_options = UniformPolicyOverOptions(rng, args.noptions)  # Fixed Uniform Pi-O

        # Different choices are possible for the critic. Here we learn an
        # option-value function and use the estimator for the values upon arrival
        option_weights = np.zeros((nfeatures, args.noptions))
        critic = IntraOptionQLearning(args.discount, args.lr_critic, option_terminations, option_weights, policy_over_options)
        #critic = IntraOptionQLearning(args.discount, args.lr_critic, option_terminations, policy_over_options.weights)

        # Learn Qomega separately
        action_weights = np.zeros((nfeatures, args.noptions, nactions))
        action_critic = IntraOptionActionQLearning(args.discount, args.lr_critic, option_terminations, action_weights, critic)

        # Improvement of the termination functions based on gradients
        termination_improvement= TerminationGradient(option_terminations, critic, args.lr_term)

        # Intra-option gradient improvement with critic estimator
        intraoption_improvement = IntraOptionGradient(option_policies, args.lr_intra)

        for episode in range(args.nepisodes):
            # if episode == 1000:
            #    env.goal = rng.choice(possible_next_goals)
            #    print('************* New goal : ', env.goal)

            return_per_episode = 0.0
            observation = env.reset()

            phi = features(observation)

            # Sample from policy over options to choose the current option
            option = policy_over_options.sample(phi)

            # For given (s,w), sample the intra-option policy to choose the action prescribed by that option.
            action = option_policies[option].sample(phi)

            critic.start(phi, option)
            action_critic.start(phi, option, action)

            cumreward = 0.
            duration = 1
            option_switches = 0
            avgduration = 0.
            for step in range(args.nsteps):
                observation, reward, done, _ = env.step(action)
                phi = features(observation)
                return_per_episode += pow(args.discount,step)*reward

                state_frequency_history[run, episode, observation, option] +=1

                # Termination might occur upon entering the new state
                # Book Keeping for Option switches, duration and average option duration
                if option_terminations[option].sample(phi):
                    option = policy_over_options.sample(phi)
                    option_switches += 1
                    avgduration += (1./option_switches)*(duration - avgduration)
                    duration = 1

                action = option_policies[option].sample(phi)

                # Critic update
                update_target = critic.update(phi, option, reward, done)
                action_critic.update(phi, option, action, reward, done)

                if isinstance(option_policies[option], SoftmaxPolicy):
                    # Intra-option policy update
                    critic_feedback = action_critic.value(phi, option, action)
                    if args.baseline:
                        critic_feedback -= critic.value(phi, option)
                    intraoption_improvement.update(phi, option, action, critic_feedback)
                    # Termination update
                    termination_improvement.update(phi, option)

                # Book Keeping: undiscounted return
                cumreward += reward
                duration += 1
                if done:
                    break

            history[run, episode, 0] = step
            history[run, episode, 1] = return_per_episode
            history[run, episode, 2] = avgduration

            print('Run {} episode {} steps {} cumreward {} avg. duration {} switches {}'.format(run, episode, step, cumreward, avgduration, option_switches))

            for o in range(args.noptions):
                weight_intra_option[run,episode, :, :, o] = option_policies[o].weights
                weight_action_value[run, episode, :, :, o] = action_critic.weights[:,o,:]
            for o in range(args.noptions):
                weight_termination[run,episode , :, o] = option_terminations[o].weights
                weight_option_value[run, episode, :, o] = critic.weights[:,o]

        np.save(os.path.join(dir_name, 'History.npy'), history)
        np.save(os.path.join(dir_name,'StateFreq.npy'), state_frequency_history)
        np.save(os.path.join(dir_name,'Weights_Termination.npy'), weight_termination)
        np.save(os.path.join(dir_name,'Weights_IntraOption.npy'), weight_intra_option)
        np.save(os.path.join(dir_name, 'Weights_OptionValueFunction.npy'), weight_option_value)     # Wts for Q(s,w)
        np.save(os.path.join(dir_name, 'Weights_ActionValueFunction.npy'), weight_action_value)  # Wts for Q(s,w,a)

































































































