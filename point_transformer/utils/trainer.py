import torch
from tqdm import tqdm

from point_transformer.dataset.shapenet_dataset import ID_TO_CLASS
from point_transformer.utils.plotters import plot_point_clouds
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
        n_classes = self.config["MODEL"]["n_classes"]

        total_intersection = torch.zeros(n_classes, device=self.device)
        total_union = torch.zeros(n_classes, device=self.device)
        val_loss = []

        pbar = tqdm(enumerate(self.val_loader), total=len(self.val_loader))
        for n_iter, (pcl, labels) in pbar:
            if n_iter > self.config["OPTIM"]["max_eval_iters"]:
                break
            pcl = pcl.to(self.device)
            labels = labels.to(self.device)

            preds = self.model(pcl, pcl)
            # Compute IoU components
            pred_labels = self.post_processor(preds)
            total_intersection, total_union = self.compute_interection_union(
                pred_labels, labels, total_intersection, total_union, n_classes
            )

            if n_iter < self.config["OPTIM"]["max_num_plots"]:
                self.plot_preds(pcl, preds, labels, iter=n_iter)

            # Compute loss
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
        miou = self.compute_miou(total_union, total_intersection, n_classes)
        self.write_float_to_tb(miou, name="val/miou", step=self.epoch)
        self.write_float_to_tb(torch.mean(torch.tensor(val_loss)).item(), name="val/avg_loss", step=self.epoch)
        pbar.close()

    def compute_interection_union(self, pred_labels, labels, total_intersection, total_union, n_classes):
        for cls in range(n_classes):
            pred_mask = pred_labels == cls
            label_mask = labels == cls
            total_intersection[cls] += (pred_mask & label_mask).sum()
            total_union[cls] += (pred_mask | label_mask).sum()
        return total_intersection, total_union

    def compute_miou(self, total_union, total_intersection, n_classes):
        ious = torch.zeros(n_classes)
        for cls in range(n_classes):
            if total_union[cls] == 0:
                ious[cls] = float("nan")
            else:
                ious[cls] = total_intersection[cls] / total_union[cls]
        miou = torch.nanmean(ious).item()
        return miou

    def post_processor(self, out):
        out = torch.softmax(out, -1)
        pred_out = torch.argmax(out, -1)
        return pred_out

    def plot_preds(self, pcl, out, labels, iter=0, batch_id=0):
        preds = self.post_processor(out)
        title = f"{ID_TO_CLASS[preds[batch_id].item()]}/{ID_TO_CLASS[labels[batch_id].item()]} - {preds[batch_id].item() == labels[batch_id].item()}"
        img = plot_point_clouds([pcl], return_figure=True, title=title)
        self.write_images_to_tb(img, self.epoch, f"img/{str(iter).zfill(4)}")
