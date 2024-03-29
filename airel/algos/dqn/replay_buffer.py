import collections
from typing import Tuple, NoReturn
import random

import torch


class ReplayBuffer:
    """Stores transitions in a buffer
    
    Parameters:
        max_size (int): maximum buffer size.When the buffer
            overflows the old memories are dropped.
        device (str): where the data will be sent. Defaults to `cpu`.
    
    Example:
    
        >>> #TODO tests

    References:
        1. `Playing Atari with Deep Reinforcement Learning <https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf>`_

    """

    def __init__(self, max_size: int, device: str = "cpu"):

        self.buffer = collections.deque(maxlen=max_size)
        self.device = device

    def __len__(self) -> int:
        return len(self.buffer)

    def store(self, transition: Tuple) -> NoReturn:
        self.buffer.append(transition)

    def sample(
            self, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Samples a given transition number uniformly.
        A transition is observation at t, action at t, reward at t,
        observation at t+1 and if the episode is finished.
        
        Parameters:
            batch_size (int): size of minibatch.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        """
        batch = random.sample(self.buffer, batch_size)
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for transition in batch:
            obs_t, action, reward, obs_tp1, done_mask = transition
            obses_t.append(obs_t)
            actions.append([action])
            rewards.append([reward])
            obses_tp1.append(obs_tp1)
            dones.append([done_mask])
        return torch.tensor(obses_t, dtype=torch.float, device=self.device), \
               torch.tensor(actions, device=self.device), \
               torch.tensor(rewards, device=self.device), \
               torch.tensor(obses_tp1, dtype=torch.float, device=self.device), \
               torch.tensor(dones, device=self.device)
