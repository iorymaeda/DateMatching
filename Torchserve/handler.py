import os
import pickle
from abc import ABC

import torch
from ts.torch_handler.base_handler import BaseHandler


class CustomHandler(BaseHandler, ABC):
    def __init__(self):
        super(CustomHandler, self).__init__()
        self.initialized = False


    def initialize(self, ctx):
        for _ in range(1000):
            print("initialize")

        print(ctx)

        self.manifest = ctx.manifest
        properties = ctx.system_properties
        model_dir = properties.get("model_dir")
        serialized_file = self.manifest["model"]["serializedFile"]
        model_pt_path = os.path.join(model_dir, serialized_file)


        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id"))
            if torch.cuda.is_available() and properties.get("gpu_id") is not None
            else "cpu"
        )


    def preprocess(self, requests) -> list:
        pass


    def inference(self, input_batch:list[torch.Tensor]) -> list[torch.Tensor]:
        pass

    def postprocess(self, inference_output:list[torch.Tensor]) -> list:
        pass