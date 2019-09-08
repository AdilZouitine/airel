import random

import numpy as np
import gym
import torch
import torch.nn.functional as F

from .. import base
from . import ReplayBuffer
from . import LinearSchedule


class DeepQLearning(base.BaseAlgo):
    """Implementation of the double Q-learning .
    
    Parameters:
        env: Gym environement.
        model: Pytorch model.
        timesteps: Number of step while training.
        gamma (float): discount factor of G_{t}=\sum_{k=t+1}^{T} \gamma^{k-t-1} R_{k}, defaults `0.99`.
        batch_size (int): Size of minibatch, defaults to `32`.
        buffer_size (int): Size of replay buffer defaults to `1e5`.
        learning_rate (float): defaults to `2.5e-4`.
        optimizer (cls): Optimizer, defaults to `torch.optim.Adam`.
        q_update_interval (int): Number of step before updating the q-network, defaults to `1`.
        target_update_interval (int): Number of step before updating the target network, defaults to `256`.
        exploration_fraction (float): fraction of entire training period over which the exploration rate is annealed.
        exploration_start (float): start value of random action probability.
        exploration_end (float): final value of random action probability.
        exploration_scheduler: exploration scheduler.
        loss: Loss function. Defaults to `smooth_l1_loss`.
        nb_update (int): Number of weight updates during an optimization phase. Defaults to `1`.
        clip_grad_norm (float): Clip the gradient of Q-network, defaults to `10`.
        seed (int): Fix the random seed.
    
    References:
        1. `Playing Atari with Deep Reinforcement Learning <https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf>`_
        2. `Deep Reinforcement Learning with Double Q-learning <https://arxiv.org/pdf/1509.06461.pdf>`_

    """

    def __init__(self,
                 env,
                 model,
                 timesteps: int,
                 learning_start: int,
                 gamma: float = 0.99,
                 batch_size: int = 32,
                 buffer_size: int = int(1e5),
                 learning_rate: float = 2.5e-4,
                 optimizer=torch.optim.Adam,
                 q_update_interval: int = 1,
                 target_update_interval: int = 256,
                 exploration_fraction: float = 0.9,
                 exploration_start: float = 1,
                 exploration_end: float = 0.05,
                 exploration_scheduler=LinearSchedule,
                 loss=F.smooth_l1_loss,
                 nb_update: int = 1,
                 clip_grad_norm: float = 10.,
                 verbose: int = 100,
                 seed: int = 42):

        # Random part
        random.seed(seed)
        
        # Environment part
        self.env = env
        self.nb_action = self.env.action_space.n

        # Model part
        self.q = model
        self.q_target = model

        # Replay buffer and exploration
        self.replay_buffer = ReplayBuffer(max_size=buffer_size)
        self.exploration_scheduler = exploration_scheduler(
            total_timesteps=timesteps,
            exploration_fraction=exploration_fraction,
            final_p=exploration_end,
            initial_p=exploration_start)
        
        # number of training iteration/update
        self.timesteps = timesteps
        self.q_update_interval = q_update_interval
        self.target_update_interval = target_update_interval
        self.learning_start = learning_start
        self.nb_update = nb_update

        # Hyperparams
        self.gamma = gamma
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer = optimizer(self.q.parameters(), lr=self.learning_rate)

        # Loss part
        self.loss = loss
        self.clip_grad_norm = clip_grad_norm

        # Logging part
        self.verbose = verbose
        self.nb_episode = 0

    def sample_action(self, obs: torch.tensor, exploration_proba: float):
        '''
        With probability \epsilon select a random action a_{t} otherwise select
        a_{t}=\max _{a} Q^{*}\left(\phi\left(s_{t}\right), a ; \theta\right)
        '''
        out = self.q(obs)
        coin = random.random()
        if coin < exploration_proba:
            return random.randint(0, self.nb_action - 1)
        else:
            return out.argmax().item()

    def _optimize(self):
        # Sample random minibatch of transitions\left(\phi_{j}, a_{j}, r_{j}, \phi_{j+1}\right)  from  Replay buffer
        obs_t, action, reward, obs_tp1, done_mask = self.replay_buffer.sample(
            self.batch_size)

        # r_{j}+\gamma \max _{a^{\prime}} Q^{\prime}\left(\phi_{j+1}, a^{\prime} ; \theta\right)
        q_out = self.q(obs_t)
        q_a = q_out.gather(1, action)
        max_q_prime = self.q_target(obs_tp1).max(1)[0].unsqueeze(1)
        target = reward + self.gamma * max_q_prime * done_mask

        # Perform a gradient descent step on Loss(\left(y_{j}-Q\left(\phi_{j}, a_{j} ; \theta\right)\right))
        loss = self.loss(q_a, target)
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.q.parameters(),
                                                   self.clip_grad_norm)
        self.optimizer.step()

    def train(self):
        done = False
        obs_t = self.env.reset()

        for step in range(self.timesteps):

            exploration_proba = self.exploration_scheduler.get(step)
            action = self.sample_action(
                obs=torch.from_numpy(obs_t).float(),
                exploration_proba=exploration_proba)

            obs_tp1, reward, done, _ = self.env.step(action)
 
            done_mask = 0.0 if done else 1.0
            self.replay_buffer.append((obs_t, action, reward, obs_tp1,
                                       done_mask))
            obs_t = obs_tp1

            if done:
                done = False
                obs_t = self.env.reset()
                self.nb_episode += 1

            if step > self.learning_start and step % self.q_update_interval == 0:
                for _ in range(self.nb_update):
                    self._optimize()

            if step % self.target_update_interval == 0 and step != 0:
                self.q_target.load_state_dict(self.q.state_dict())
