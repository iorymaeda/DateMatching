import random

from torch.utils.data import Dataset


class EmbedDataset(Dataset):
    def __init__(self, x, y,
                 decompositor=None,
                 **kwargs):
        assert len(x) == len(y)
        
        self.x_opposite = x[y[:, 0] == 0.]
        self.x_target = x[y[:, 0] == 1.]
        self.y_opposite = y[y[:, 0] == 0.]
        self.y_target = y[y[:, 0] == 1.]
        self.decompositor = decompositor
        
        
    def __len__(self):
        return len(self.y_opposite) + len(self.y_target)
    
    
    def __getitem__(self, _):
        if random.random() > 0.5:
            idx = random.choice(range(len(self.x_opposite)))
            x, y = self.x_opposite[idx], self.y_opposite[idx]
            
        else:
            idx = random.choice(range(len(self.x_target)))
            x, y = self.x_target[idx], self.y_target[idx]
            
        if self.decompositor:
            x = self.decompositor(x).astype('float32')
            
        return x, y


class TestDataset(Dataset):
    def __init__(self, x, y,
                 decompositor=None,
                 **kwargs):
        assert len(x) == len(y)
        
        self.x = x
        self.y = y
        self.decompositor = decompositor
        
        
    def __len__(self):
        return len(self.y)
    
    
    def __getitem__(self, idx):
        y = self.y[idx]
        x = self.x[idx]
        
        if self.decompositor:
            x = self.decompositor(x).astype('float32')
            
        return x, y


