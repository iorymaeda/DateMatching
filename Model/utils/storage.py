import pickle
import numpy as np


class EmbedingStorage:
    """This storage class contains face id and embeddings
    All prepared photos storage in `data/target/` or `data/opposite` with unique id.
    useful for dataset modification and handmade markups - just move photos from opposite to target or from target to opposite
    """
    def __init__(self, path=None):
        self.path = path
        self.id_path: dict[int, str] = {} # id -> path
        self.id_embs: dict[int, np.ndarray] = {} # id -> face embedding
        
        self.id_pid: dict[int, int] = {} # id -> person id 
        self.name_pid: dict[int, int] = {} # person name -> person id
        self.pid_name: dict[int, int] = {} # person id -> person name
        
        
        
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
                'id_path': self.id_path,
                'id_embs': self.id_embs,
                'id_pid': self.id_pid,
                'name_pid': self.name_pid,
                'pid_name': self.pid_name,
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
            self.id_path = loaded_dict['id_path']
            self.id_embs = loaded_dict['id_embs']
            self.id_pid = loaded_dict['id_pid']
            self.name_pid = loaded_dict['name_pid']
            self.pid_name = loaded_dict['pid_name']
            
            
    def __generate_emb_id(self) -> int:
        return len(self.id_path)
    
    
    def __generate_person_id(self, person:str) -> int:
        person = person.lower()
        
        if person in self.name_pid:
            return self.name_pid[person]
        else:
            return len(self.name_pid)
    
    
    def __setitem__(self, key: str, value: list[np.ndarray, str]):
        """Put embeddings in storage
        :param key: primary photo path+name
        :param value: list with embeddings from models and person name
            `noname` if there no name
        """
        
        assert isinstance(value, list)
        if len(value) == 1: value.append('noname')
        assert len(value) == 2
        assert isinstance(key, str)
        assert isinstance(value, list)
        
        emb, name = value
        assert isinstance(emb, np.ndarray)
        assert isinstance(name, str)
        
        eid = self.__generate_emb_id()
        pid = self.__generate_person_id(name)
        
        self.cached_id = eid
        
        self.id_path[eid] = key
        self.id_embs[eid] = emb
        
        self.id_pid[eid] = pid
        self.pid_name[pid] = name
        self.name_pid[name] = pid
    
    
    def __getitem__(self, item: int) -> np.ndarray:
        """Get embeddings from storage
        :param key: id 
        """
        if isinstance(item, int):
            return self.id_embs[item]
        
        elif isinstance(item, slice):
            _arr = np.concatenate(list(self.id_embs.values()))
            return _arr[item]
        
        else:
            raise TypeError(f'Dont understand key type: {type(item)}')
            

    def __len__(self) -> int:
        return len(self.id_path)
    
    
    def __repr__(self):
        msg = f"number of semples: {len(self)}\nnumber of persons: {len(self.name_pid)}"
        return msg


    def __str__(self):
        msg = f"number of semples: {len(self)}\nnumber of persons: {len(self.name_pid)}"
        return msg
