import torch
from torch.utils.data import Dataset


class Memory(Dataset):
    def __init__(self, states, actions, log_probs, rewards, advantages, values) -> None:
        super().__init__()

        self.states = states
        self.actions = actions
        self.log_probs = log_probs
        self.rewards = rewards
        self.advantages = advantages
        self.values = values

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return (
            self.states[idx],
            self.actions[idx],
            self.log_probs[idx],
            self.rewards[idx],
            self.advantages[idx],
            self.values[idx],
        )
