import torch.nn as nn


class BaseModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.img_size = config["DATA"]["img_size"]


class TemplateModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.l1 = nn.Linear(1, 1)

    def forward(self, x):
        return x
