import random

from torch.utils.data import Dataset


class EmbedDataset(Dataset):
    def __init__(self, x, y, embeder,
                 decompositor=None,
                 **kwargs):
        assert len(x) == len(y)
        
        self.embeder = embeder
        self.x_opposite = x[y == 0.]
        self.x_target = x[y == 1.]
        self.y_opposite = y[y == 0.]
        self.y_target = y[y == 1.]
        self.decompositor = decompositor
        
        
    def __len__(self):
        return len(self.y_opposite) + len(self.y_target)
    
    
    def __getitem__(self, _):
        if random.random() > 0.5:
            target = False
            idx = random.choice(range(len(self.x_opposite)))
            xid, y = self.x_opposite[idx], self.y_opposite[idx]
            
        else:
            target = True
            idx = random.choice(range(len(self.x_target)))
            xid, y = self.x_target[idx], self.y_target[idx]
            
            
        x = self.embeder[int(xid)]
        if self.decompositor:
            x = self.decompositor(x).astype('float32')
        
        return x[0], y[None, ...]


class TestDataset(Dataset):
    def __init__(self, x, y, embeder,
                 decompositor=None,
                 **kwargs):
        assert len(x) == len(y)
        
        self.x = x
        self.y = y

        self.embeder = embeder
        self.decompositor = decompositor
        
        
    def __len__(self):
        return len(self.y)
    
    
    def __getitem__(self, idx):
        y = self.y[idx]
        x = self.embeder[int(self.x[idx])]
        
        if self.decompositor:
            x = self.decompositor(x).astype('float32')
            
        return x[0], y[None, ...]