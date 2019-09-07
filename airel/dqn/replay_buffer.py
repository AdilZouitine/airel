import collections
from typing import Tuple, NoReturn
import random

import torch


class ReplayBuffer:
    """Stores transitions in a buffer
    
    Parameters:
        max_size (int): maximum buffer size
    
    Example:
    
        >>> #TODO tests
    """

    def __init__(self, max_size: int):
        self.buffer = collections.deque(maxlen=max_size)

    def __len__(self) -> int:
        return len(self.buffer)

    def append(self, transition: Tuple) -> NoReturn:
        self.buffer.append(transition)

    def sample(
            self, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Samples a given transition number uniformly.
        A transition is observation at t, action at t, reward at t,
        observation at t+1 and if the episode is finished.
        
        Arguments:
            batch_size {int} -- [description]
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] -- [description]
        """
        batch = random.sample(self.buffer, batch_size)
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for transition in batch:
            obs_t, action, reward, obs_tp1, done = transition
            obses_t.append(obs_t)
            actions.apprend([action])
            rewards.append([reward])
            obses_tp1.append(obs_tp1)
            dones.append([dones])

        return torch.tensor(obses_t, dtype=torch.float), torch.tensor(actions), \
               torch.tensor(rewards), torch.tensor(obses_tp1, dtype=torch.float), \
               torch.tensor(dones)