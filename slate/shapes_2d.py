import numpy as np
from torch.utils.data import Dataset
import h5py

class CswmStyleDataset(Dataset):

    def __init__(self, root, phase, traj_size, reward=False):
        self.root = root
        self.reward = reward
        self.traj_size = traj_size
        with h5py.File(root, 'r') as f:
            self.keys = list(f.keys())
            part_1 = int(len(self.keys) * 0.8)
            part_2 = int(len(self.keys) * 0.9)
            if phase == 'train':
                self.start = 0
                self.len = part_1-self.start
            elif phase == 'val':
                self.start = part_1
                self.len = part_2 - self.start
            elif phase == 'test':
                self.start = part_2
                self.len = len(self.keys) - self.start
            else:
                raise NotImplementedError

    def __getitem__(self, index):
        with h5py.File(self.root, 'r') as f:
            img = np.array(f[str(int(self.start+index))]['obs'][:self.traj_size])
            action = np.array(f[str(int(self.start+index))]['action'][:self.traj_size])
            if self.reward:
                reward = np.sum(np.array(f[str(int(self.start+index))]['reward'][:self.traj_size]))[None] # TODO discount
                return img, action, reward
            return img, action

    def __len__(self):
        return self.len