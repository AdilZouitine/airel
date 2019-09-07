import gym
import torch

from .. import base
from . import ReplayBuffer 

class DeepQLearning(base.BaseAlgo):
    """[summary]
    
    Parameters:
        env: Gym environement.
        model: Pytorch model.
        timestep: Number of step while training.
        discount (float): discount factor of $G_{t}=\sum_{k=t+1}^{T} \gamma^{k-t-1} R_{k}$, defaults `0.99`.
        batch_size (int): Size of minibatch, defaults to `32`.
        buffer_size (int): Size of replay buffer defaults to `1e5`.
        learning_rate (float): defaults to `2.5e-4`.
        optimizer (cls): Optimizer, defaults to `torch.optim.Adam`.
        target_update_interval (int): Number of step before updating target network, defaults to `256`.
    """
    
    def __init__(self,
                 env,
                 model,
                 timestep:int,
                 learning_start:int,
                 discount=0.99:float,
                 batch_size=32:int,
                 buffer_size=int(1e5):int, 
                 learning_rate=2.5e-4:float,
                 optimizer=torch.optim.Adam,
                 target_update_interval=256:int,
                 exploration_fraction=0.9:float,
                 exploration_start=100:int,
                 exploration_end=5:int):

        self.env = env
        self.q = model
        self.q_target = model.clone()
        self.timestep = timestep
        self.learning_start = learning_start,
        self.discount = discount
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer = optimizer(self.q.parameters, lr=learning_rate)
        self.target_update_interval=target_update_interval
        