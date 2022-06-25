"""Handler for torchserve
ref: https://github.com/pytorch/serve/blob/master/ts/torch_handler/base_handler.py"""

import os
import pickle
from abc import ABC

import torch
from ts.torch_handler.base_handler import BaseHandler


class CustomHandler(BaseHandler, ABC):
    def __init__(self):
        super(CustomHandler, self).__init__()
        self.initialized = False
        raise NotImplementedError


    def initialize(self, ctx):
        pass


    def preprocess(self, requests) -> list:
        pass


    def inference(self, input_batch:list[torch.Tensor]) -> list[torch.Tensor]:
        pass

    def postprocess(self, inference_output:list[torch.Tensor]) -> list:
        pass