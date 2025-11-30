import torch
from tqdm import tqdm

from point_transformer.utils.trainer_base import TrainerBase


class Trainer(TrainerBase):
    def __init__(self, config, logger):
        super().__init__(config, logger)

    def train(self):
        self.logger.info("Started training..")
        start_epoch = self.epoch
        for epoch in range(start_epoch, self.config["OPTIM"]["num_epochs"]):
            self.epoch = epoch
            self.evaluate_model()
            self.train_one_epoch()
            if epoch % self.config["OPTIM"]["save_ckpt_every"] == 0:
                self.save_checkpoint()

    def train_one_epoch(self):
        self.model.train()
        pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
        for n_iter, (pcl, labels) in pbar:
            pcl = pcl.to(self.device)
            labels = labels.to(self.device)

            preds = self.model(pcl, pcl)
            loss, loss_dict = self.loss_fn(preds, labels)
            self.write_dict_to_tb(loss_dict, self.total_iters_train, prefix="train")

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.total_iters_train += 1
            pbar.set_postfix(
                {
                    "mode": "train",
                    "epoch": f"{self.epoch}/{self.config['OPTIM']['num_epochs']}",
                    "loss": loss.item(),
                }
            )
        pbar.close()

    @torch.no_grad()
    def evaluate_model(self, save_plots=False):
        self.model.eval()
        pbar = tqdm(enumerate(self.val_loader), total=len(self.val_loader))
        val_loss = []
        for n_iter, (pcl, labels) in pbar:
            if n_iter > self.config["OPTIM"]["max_eval_iters"]:
                break
            pcl = pcl.to(self.device)
            labels = labels.to(self.device)

            preds = self.model(pcl, pcl)
            loss, loss_dict = self.loss_fn(preds, labels)
            val_loss.append(loss)
            self.write_dict_to_tb(loss_dict, self.total_iters_val, prefix="val")
            self.total_iters_val += 1
            pbar.set_postfix(
                {
                    "mode": "val",
                    "epoch": f"{self.epoch}/{self.config['OPTIM']['num_epochs']}",
                    "loss": loss.item(),
                }
            )
        self.write_float_to_tb(torch.mean(torch.tensor(val_loss)).item(), name="val/avg_loss", step=self.epoch)
        pbar.close()

    def post_processor(self, out):
        out = torch.softmax(out, -1)
        pred_out = torch.argmax(out, -1)
        return pred_out
