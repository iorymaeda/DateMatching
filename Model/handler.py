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


    def initialize(self, ctx):
        print('initialize')

    def preprocess(self, requests) -> list:
        print('preprocess')


    def inference(self, input_batch:list[torch.Tensor]) -> list[torch.Tensor]:
        print('inference')

    def postprocess(self, inference_output:list[torch.Tensor]) -> list:
        print('postprocess')