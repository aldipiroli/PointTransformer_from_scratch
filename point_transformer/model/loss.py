import torch.nn as nn


class BaseLoss(nn.Module):
    def __init__(self, config, logger):
        super(BaseLoss, self).__init__()
        self.config = config
        self.logger = logger

    def forward(self, preds, labels):
        pass


class ClsLoss(BaseLoss):
    def __init__(self, config, logger):
        super(ClsLoss, self).__init__(config, logger)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, preds, labels):
        loss = self.loss_fn(preds, labels)

        loss_dict = {}
        loss_dict["cls_loss"] = loss
        return loss, loss_dict


class SegmLoss(BaseLoss):
    def __init__(self, config, logger):
        super(SegmLoss, self).__init__(config, logger)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, preds, labels):
        loss = self.loss_fn(preds.permute(0, 2, 1), labels)

        loss_dict = {}
        loss_dict["segm_loss"] = loss
        return loss, loss_dict
