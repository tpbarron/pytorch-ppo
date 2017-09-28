import argparse
import sys
import math
from collections import namedtuple
from itertools import count

import gym
import numpy as np

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from torch.autograd import Variable
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from storage import RolloutStorage
from models import DiscretePolicy, Value, DiscreteActorCritic, DiscreteConvActorCritic, ContinuousActorCritic #Policy, Value, ActorCritic
from running_state import ZFilter
import envs

PI = torch.FloatTensor([3.1415926])
E = torch.FloatTensor([2.7182818])
parser = argparse.ArgumentParser(description='PyTorch Proximal Policy Optimization')
parser.add_argument('--gamma', type=float, default=0.995, metavar='G',
                    help='discount factor (default: 0.995)')
parser.add_argument('--env-name', default="LunarLanderContinuous-v2", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--tau', type=float, default=0.95, metavar='G',
                    help='gae (default: 0.97)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--sample-batch-size', type=int, default=20, metavar='N',
                    help='batch size (default: 20)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=25, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--entropy-coeff', type=float, default=0.01, metavar='N',
                    help='coefficient for entropy cost')
parser.add_argument('--clip-epsilon', type=float, default=0.1, metavar='N',
                    help='Clipping for PPO grad')
parser.add_argument('--use-joint-pol-val', action='store_true',
                    help='whether to use combined policy and value nets')
parser.add_argument('--epochs-per-batch', type=int, default=1,
                    help='number of passes though the sampled data')
parser.add_argument('--max-episode-steps', type=int, default=1000,
                    help='Maximum number of steps in an episode')
args = parser.parse_args()

# env = envs.make_env(args.env_name, args.seed, ".")
env = gym.make(args.env_name)
num_inputs = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]
print ("num : ", num_actions)
print (env.action_space.shape, env.action_space.low, env.action_space.high)
env.seed(args.seed)
torch.manual_seed(args.seed)

if args.use_joint_pol_val:
    ac_net = ContinuousActorCritic(num_inputs, num_actions)
    opt_ac = optim.Adam(ac_net.parameters(), lr=0.001)
# else:
#     policy_net = DiscretePolicy(num_inputs, num_actions)
#     value_net = Value(num_inputs)
#     opt_policy = optim.Adam(policy_net.parameters(), lr=0.001)
#     opt_value = optim.Adam(value_net.parameters(), lr=0.001)

def normal_log_density(x, mean, log_std, std):
    var = std.pow(2)
    log_density = -(x - mean).pow(2) / (2 * var) - 0.5 * (torch.log(2 * Variable(PI) * std))
    return log_density.sum(1)

# def normal_pdf(x, mean, std):
#     return 1. / (torch.sqrt(2. * Variable(PI) * std.pow(2.))) * torch.exp(-(x - mean).pow(2) / (2 * std.pow(2.)))
#
# def log_normal_pdf(x, mean, std):
#     return torch.log(normal_pdf(x, mean, std))

def ppo_update():
    advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
    advantages = (advantages - advantages.mean()) / advantages.std()
    for _ in range(args.epochs_per_batch):
        sampler = BatchSampler(SubsetRandomSampler(range(args.sample_batch_size)), args.sample_batch_size/10, drop_last=False)
        for indices in sampler:
            # Reshape to do in a single forward pass for all steps
            states_batch = rollouts.states[:-1][indices].view(-1, *rollouts.states.size()[-1:])
            actions_batch = rollouts.actions.view(-1, 1)[indices]
            return_batch = rollouts.returns[:-1].view(-1, 1)[indices]

            action_mean, action_log_std, action_std, values = ac_net(Variable(states_batch))
            # action_log_probs = log_normal_pdf(Variable(actions_batch).float(), action_mean, action_std)
            action_log_probs = normal_log_density(Variable(actions_batch).float(), action_mean, action_log_std, action_std)

            old_action_mean = rollouts.old_action_means.view(-1, rollouts.old_action_means.size(-1))[indices]
            old_action_log_std = rollouts.old_action_log_stds.view(-1, rollouts.old_action_log_stds.size(-1))[indices]
            old_action_std = rollouts.old_action_stds.view(-1, rollouts.old_action_stds.size(-1))[indices]
            old_action_log_probs = normal_log_density(Variable(actions_batch).float(),
                                                      Variable(old_action_mean, volatile=True),
                                                      Variable(old_action_log_std, volatile=True),
                                                      Variable(old_action_std, volatile=True)).data
            # print ("old act log probs: ", old_action_log_probs.size())
            # input("")

            ratio = torch.exp(action_log_probs - Variable(old_action_log_probs))
            adv_targ = Variable(advantages.view(-1, 1)[indices])
            surr1 = ratio * adv_targ
            surr2 = ratio.clamp(1.0 - args.clip_epsilon, 1.0 + args.clip_epsilon) * adv_targ
            action_loss = -torch.min(surr1, surr2).mean() # PPO's pessimistic surrogate (L^CLIP)

            dist_entropy = (-0.5 * torch.log(2.0 * Variable(PI) * Variable(E) * action_std.pow(2))).mean()

            value_loss = (Variable(return_batch) - values).pow(2).mean()

            opt_ac.zero_grad()
            (value_loss + action_loss - dist_entropy * args.entropy_coeff).backward()
            opt_ac.step()

from visdom import Visdom
vis = Visdom()

# reward first point to append to
win = vis.line(
    X=np.array([0]),
    Y=np.array([0]),
    opts=dict(title=args.env_name, xaxis={'title':'episode'}, yaxis={'title':'reward'})
)

# initial reset, will run continuously from now on
obs = env.reset()
print ("Obs: ", obs.shape)
episode_reward = 0.
episode_step = 0
episode_num = 0

obs_shape = env.observation_space.shape
action_shape = env.action_space.shape
rollouts = RolloutStorage(args.sample_batch_size, obs_shape, action_shape)
current_state = torch.zeros(1, *obs_shape)
def update_current_state(state):
    state = torch.from_numpy(state).float()
    current_state = state
    return current_state

state = env.reset()
current_state = update_current_state(state)
rollouts.states[0].copy_(current_state)
episode_reward = 0
episode = 0

for i_update in count(1):
    for step in range(args.sample_batch_size):

        # Sample actions
        state_var = Variable(rollouts.states[step], volatile=True)
        action_mean, action_log_std, action_std, value = ac_net(state_var)
        action = torch.normal(action_mean, action_std)
        # log_probs = normal_log_density(action, action_mean, action_log_std, action_std).data
        action = action.data

        action_np = action.numpy()[0]
        # print ("orig act: ", action_np)
        action_np = np.clip(action_np, env.action_space.low, env.action_space.high)
        # print ("Clipped act: ", action_np)
        # if args.render or episode % args.log_interval == 0:
        #     env.render()

        # Obser reward and next state
        state, reward, done, info = env.step(action_np)
        if np.isnan(reward):
            print ("ISNAN")
            print (reward, action, action_np, done, action_mean, action_std, state_var)
            input("")
            reward = 0.0
        episode_reward += reward
        reward = torch.FloatTensor([reward])

        # If done then clean the history of observations.
        masks = torch.FloatTensor([0.0]) if done else torch.FloatTensor([1.0])

        if done:
            episode += 1
            state = env.reset()
            if episode % args.log_interval == 0:
                print ("episode_reward = ", episode_reward)
            vis.line(
                X=np.array([episode]),
                Y=np.array([episode_reward]),
                win=win,
                update='append'
            )
            episode_reward = 0

        current_state = update_current_state(state)
        rollouts.insert(step, current_state, action, value.data, action_mean.data, action_log_std.data, action_std.data, reward, masks)

    next_value = ac_net(Variable(rollouts.states[-1], volatile=True))[3].data
    rollouts.compute_returns(next_value, True, args.gamma, args.tau)
    ppo_update()

    rollouts.states[0].copy_(rollouts.states[-1])
