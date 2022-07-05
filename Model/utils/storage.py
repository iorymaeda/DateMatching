import pickle

import torch
import numpy as np



class PhotoEmbedingStorage:
    """This storage class contains:
    generated unique id: list of embeddings
    generated unique id: list of face detected information

    All prepared photos storage in `data/outputs/` with unique id.
    All raw photos storage in `data/target/` or `data/opposite` with unique id
    useful for dataset modification and handmade markups - just move photos from opposite to target or from target to opposite
    """
    def __init__(self, path=None):
        self.path = path
        self.emb_num: int = 0
        self.path_id: dict[str, int] = {}
        self.id_path: dict[int, str] = {}
        self.id_embs: dict[int, np.ndarray] = {}
        
        if path:
            self.load()


    def save(self, path=None):
        "Save storage to path"
        if path is None:
            if self.path:
                path = self.path
            else:
                raise Exception('Provide path to save')

        with open(path, 'wb') as f:
            pickle.dump({
                'path_id': self.path_id,
                'id_path': self.id_path,
                'id_embs': self.id_embs,
                'emb_num': self.emb_num,
            }, f)
            
                
    def load(self, path=None):
        "Load storage to path"
        if path is None:
            if self.path:
                path = self.path
            else:
                raise Exception('Provide path to save')
    
        with open(path, 'rb') as f:
            loaded_dict = pickle.load(f)
            self.path_id = loaded_dict['path_id']
            self.id_path = loaded_dict['id_path']
            self.id_embs = loaded_dict['id_embs']
            self.emb_num = loaded_dict['emb_num']
            
            
    def __generate_id(self) -> int:
        return len(self.path_id)
    
    
    def __setitem__(self, key: str, value: list[np.ndarray|torch.Tensor]):
        """Put embeddings in storage
        :param key: primary photo path+name
        :param value: list with embeddings from models
        """
        assert isinstance(key, str)
        assert isinstance(value, list)
        
        for emb in value:
            assert isinstance(emb, (np.ndarray, torch.Tensor))
        
        # Check how many we have embeddings per sample
        if self.emb_num:
            assert len(value) == self.emb_num
        else:
            self.emb_num = len(value)
            
        uid = self.__generate_id()
        self.path_id[key] = uid
        self.id_path[uid] = key
        self.id_embs[uid] = value
        
    
    def __getitem__(self, item: tuple[int|str, int]) -> np.ndarray:
        """Get embeddings from storage
        if item is tuple, there are key and n
        :param key: id or primary path
        :param n: since we have different embeddings
            from different models, this argument responsible 
            for number of returned embedding, N - embedding from N+1 model.
            default: 0
        """
        if isinstance(item, tuple):
            key, n = item
        else:
            key, n = item, 0
        
        if isinstance(key, int):
            return self.id_embs[key][n]
        
        elif isinstance(key, str):
            uid = self.path_id[key]
            return self.id_embs[uid][n]
        
        else:
            raise TypeError(f'Dont understand key type: {type(key)}')
            

    def __len__(self) -> int:
        return len(self.path_id)
    
    
    def __repr__(self):
        msg = f"number of semples: {len(self)}\n"
        msg+= f"number of embeddings: {self.emb_num}" 
        return msg


    def __str__(self):
        msg = f"number of semples: {len(self)}\n"
        msg+= f"number of embeddings: {self.emb_num}" 
        
        if self.path_id:
            msg+= "\n"
            
        for key in self.path_id:
            msg+= f"{self.path_id[key]}: {key}"
            msg+= "\n"
        return msg


