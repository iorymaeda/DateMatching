import torch
import torch.nn as nn


class Trainer:
    def __init__(self, 
        model, loss_fn, optimizer, 
        stop_batch, metric=None,
        device='cuda', fp16=False, 
        **kwargs):
        self.model: nn.Module = model
        self.device = device
        self.metric = metric
        
        self.stop_batch = stop_batch
        self.loss_fn = loss_fn
        self.optimizer = optimizer  
        
        self.fp16 = fp16
        if fp16:
            self.scaler = torch.cuda.amp.GradScaler()        
        
        
    def checkpoint(self) -> dict:
        cpoint =  {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        if self.fp16:
            cpoint["scaler"] = self.scaler.state_dict()
            
        return cpoint
    
        
    def train(self, dataset, epoch) -> float:
        self.model.train()
        running_loss = 0
        for idx, batch in enumerate(dataset):
            X, Y = batch
            X, Y = X.to(self.device), Y.to(self.device)
            running_loss+= self.__train(X, Y)
            if idx >= self.stop_batch:
                break
                
        return running_loss
            
            
    def val(self, dataset):
        """Predict dataset
        :param dataset: dataset to predict 
        :type dataset: torch.utils.data.DataLoader
        :returns: tuple with two torch.Tensor pred, true"""
        self.model.eval()
        val_pred, val_true = [], []
        with torch.inference_mode():
            for batch in dataset:
                X, Y = batch
                val_pred+= [self.model(X.to(self.device)).cpu()]
                val_true+= [Y]
        return torch.cat(val_pred), torch.cat(val_true)
                
            
    def __train(self, X, Y) -> float:
        self.optimizer.zero_grad()
        if self.fp16:
            with torch.cuda.amp.autocast(enabled=True):
                outputs = self.model(X)
                loss = self.loss_fn(outputs, Y)
                
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

        else:
            outputs = self.model(X)
            loss = self.loss_fn(outputs, Y)    
            loss.backward()
            self.optimizer.step()
            
        if self.metric:
            self.metric(outputs.sigmoid(), Y.int())
            
        return loss.item()