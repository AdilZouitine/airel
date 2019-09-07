import random

import gym
import torch
import torch.nn.functional as F

from .. import base
from . import ReplayBuffer 
from . import LinearSchedule

class DeepQLearning(base.BaseAlgo):
    """[summary]
    
    Parameters:
        env: Gym environement.
        model: Pytorch model.
        timesteps: Number of step while training.
        discount (float): discount factor of $G_{t}=\sum_{k=t+1}^{T} \gamma^{k-t-1} R_{k}$, defaults `0.99`.
        batch_size (int): Size of minibatch, defaults to `32`.
        buffer_size (int): Size of replay buffer defaults to `1e5`.
        learning_rate (float): defaults to `2.5e-4`.
        optimizer (cls): Optimizer, defaults to `torch.optim.Adam`.
        target_update_interval (int): Number of step before updating target network, defaults to `256`.
        exploration_fraction (float): fraction of entire training period over which the exploration rate is annealed.
        exploration_start (float): start value of random action probability.
        exploration_end (float): final value of random action probability.
        exploration_scheduler: exploration scheduler.
    """
    
    def __init__(self,
                 env,
                 model,
                 timesteps:int,
                 learning_start:int,
                 discount=0.99:float,
                 batch_size=32:int,
                 buffer_size=int(1e5):int, 
                 learning_rate=2.5e-4:float,
                 optimizer=torch.optim.Adam,
                 target_update_interval=256:int,
                 exploration_fraction=0.9:float,
                 exploration_start=1:float,
                 exploration_end=0.05:FloatingPointError,
                 exploration_scheduler=LinearSchedule,
                 loss=F.smooth_l1_loss):

        self.env = env
        self.nb_action = self.env.action_space.n
        self.q = model
        self.replay_buffer = ReplayBuffer(max_size=buffer_size)
        self.q_target = model.clone()
        self.timesteps = timesteps
        self.learning_start = learning_start,
        self.discount = discount
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer = optimizer(self.q.parameters, lr=learning_rate)
        self.target_update_interval=target_update_interval
        self.exploration_scheduler = exploration_scheduler(total_timesteps=self.timestep,
                                                           exploration_fraction=exploration_fraction,
                                                           final_p=exploration_end,
                                                           initial_p=exploration_start)
        self.loss = loss
        self.nb_episode = 0
        
        def sample_action(self, obs: Torch.tensor, exploration_proba:float):
            out = self.q(obs)
            coin = random.random()
            if coin < exploration_proba:
                return random.randint(0,self.nb_action -1)
            else : 
                return out.argmax().item()
        
            
    
        def train(self):

            done = False
            obs_t = self.env.reset()
            for step in range(self.timesteps):
                
                exploration_proba = self.exploration_scheduler.get(step)
                action = self.sample_action(torch.from_numpy(obs_t).float(), exploration_proba)
                obs_tp1, reward, done, _ = env.step(action)
                self.replay_buffer.append(obs_t, action, reward, obs_tp1, done)

                if done:
                    done = False
                    obs = self.env.reset()
                
                obs_t = obs_tp1
                
                if step > self.learning_start:
                    obs_t, action, reward, obs_tp1, done = self.replay_buffer.sample(self.batch_size)
                    q_out = self.q(obs_t)
                    q_a = q_out.gather(1, action)
                    max_q_prime = self.q_target(obs_tp1).max(1)[0].unsqueeze(1)
                    target = reward + self.gamma * max_q_prime * done
                    loss = self.loss(q_a, target)
                    
                    self.optimizer.zero_grad()
                    self.loss.backward()
                    self.optimizer.step()