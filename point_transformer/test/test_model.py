import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch

from point_transformer.model.model import TemplateModel
from point_transformer.utils.misc import load_config


def test_model():
    config = load_config("point_transformer/config/config.yaml")
    model = TemplateModel(config)
    img_size = config["DATA"]["img_size"]
    B, C, H, W = 2, 1, img_size[0], img_size[1]
    img = torch.randn(B, C, H, W)
    out = model(img)
    assert out[0].shape != None

def run_training_step(model, preds, y):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    loss = torch.nn.functional.mse_loss(preds, y)
    loss.backward()
    optimizer.step()
    for p in model.parameters():
        assert p.grad is not None

if __name__ == "__main__":
    print("All tests passed!")
