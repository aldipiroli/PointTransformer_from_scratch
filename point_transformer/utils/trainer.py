import torch
from torchmetrics import Accuracy, JaccardIndex
from tqdm import tqdm

from point_transformer.dataset.shapenet_dataset import ID_TO_CLASS
from point_transformer.utils.plotters import plot_cls_preds, plot_semseg_preds
from point_transformer.utils.trainer_base import TrainerBase


class Trainer(TrainerBase):
    def __init__(self, config, logger):
        super().__init__(config, logger)
        self.init_metrics()

    def init_metrics(self):
        num_classes = self.config["MODEL"]["n_classes"]
        self.metric_acc = Accuracy(task="multiclass", num_classes=num_classes).to(self.device)
        self.metric_iou = JaccardIndex(task="multiclass", num_classes=num_classes).to(self.device)

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
        self.metric_acc.reset()
        self.metric_iou.reset()
        losses = []

        pbar = tqdm(enumerate(self.val_loader), total=len(self.val_loader))
        for n_iter, (pcl, labels) in pbar:
            if n_iter > self.config["OPTIM"]["max_eval_iters"]:
                break
            pcl = pcl.to(self.device)
            labels = labels.to(self.device)

            preds = self.model(pcl, pcl)
            pred_labels = self.post_processor(preds)
            self.update_metrics(pred_labels, labels)

            loss, loss_dict = self.loss_fn(preds, labels)
            losses.append(loss.item())
            self.write_dict_to_tb(loss_dict, self.total_iters_val, prefix="val")
            self.total_iters_val += 1

            if n_iter < self.config["OPTIM"]["max_num_plots"]:
                if self.config["MODEL"]["type"] == "classification":
                    self.plot_preds_cls(pcl, preds, labels, iter=n_iter)
                else:
                    self.plot_preds_segm(pcl, preds, labels, iter=n_iter)

            pbar.set_postfix(
                {
                    "mode": "val",
                    "epoch": f"{self.epoch}/{self.config['OPTIM']['num_epochs']}",
                    "loss": loss.item(),
                }
            )

        val_loss = torch.mean(torch.tensor(losses)).item()
        self.write_float_to_tb(val_loss, name="val/val_loss", step=self.epoch)

        accuracy, miou = self.metric_computer()
        self.write_float_to_tb(accuracy, name="val/accuracy", step=self.epoch)
        self.write_float_to_tb(miou, name="val/miou", step=self.epoch)
        self.logger.info(f"Epoch: {self.epoch}, val_loss: {val_loss},  accuracy: {accuracy}, miou: {miou}")
        pbar.close()

    def update_metrics(self, pred_labels, labels):
        self.metric_acc.update(pred_labels, labels)
        if self.config["MODEL"]["type"] == "segmentation":
            self.metric_iou.update(pred_labels, labels)

    def metric_computer(self):
        accuracy = self.metric_acc.compute().item()
        if self.config["MODEL"]["type"] == "segmentation":
            miou = self.metric_iou.compute().item()
        else:
            miou = -1
        return accuracy, miou

    def compute_interection_union(self, pred_labels, labels, total_intersection, total_union, n_classes):
        for cls in range(n_classes):
            pred_mask = pred_labels == cls
            label_mask = labels == cls
            total_intersection[cls] += (pred_mask & label_mask).sum()
            total_union[cls] += (pred_mask | label_mask).sum()
        return total_intersection, total_union

    def post_processor(self, out):
        out = torch.softmax(out, -1)
        pred_out = torch.argmax(out, -1)
        return pred_out

    def plot_preds_cls(self, pcl, out, labels, iter=0, batch_id=0):
        preds = self.post_processor(out)
        if self.config["MODEL"]["type"] == "classification":
            title = f"{ID_TO_CLASS[preds[batch_id].item()]}/{ID_TO_CLASS[labels[batch_id].item()]} - {preds[batch_id].item() == labels[batch_id].item()}"
        else:
            title = None
        img = plot_cls_preds([pcl], return_figure=True, title=title)
        self.write_images_to_tb(img, self.epoch, f"img/{str(iter).zfill(4)}")

    def plot_preds_segm(self, pcl, out, labels, iter=0, batch_id=0):
        preds = self.post_processor(out)
        img = plot_semseg_preds(
            [pcl[batch_id], pcl[batch_id]], [labels[batch_id], preds[batch_id]], return_figure=False
        )
        self.write_images_to_tb(img, self.epoch, f"img/{str(iter).zfill(4)}")


###########################################
import debugpy

debugpy.listen(("localhost", 6001))
print("Waiting for debugger attach...")
debugpy.wait_for_client()
###########################################
