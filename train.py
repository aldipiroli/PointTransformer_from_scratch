import argparse

from point_transformer.dataset.shapenet_dataset import ShapeNetDataset
from point_transformer.model.loss import ClsLoss, SegmLoss
from point_transformer.model.model import PointTransformerClassification, PointTransformerSemanticSegmentation
from point_transformer.utils.misc import get_logger, load_config, make_artifacts_dirs
from point_transformer.utils.trainer import Trainer


def train(args):
    config = load_config(args.config)
    config = make_artifacts_dirs(config, log_datetime=True)
    logger = get_logger(config["LOG_DIR"])
    trainer = Trainer(config, logger)

    train_dataset = ShapeNetDataset(cfg=config, mode="train", logger=logger)
    val_dataset = ShapeNetDataset(cfg=config, mode="val", logger=logger)

    if config["MODEL"]["type"] == "classification":
        model = PointTransformerClassification(config)
        loss = ClsLoss(config, logger)
    else:
        model = PointTransformerSemanticSegmentation(config)
        loss = SegmLoss(config, logger)

    trainer.set_model(model)
    if args.ckpt is not None:
        trainer.load_checkpoint(args.ckpt)

    trainer.set_dataset(train_dataset, val_dataset, data_config=config["DATA"], val_set_batch_size=1)
    trainer.set_optimizer(optim_config=config["OPTIM"])
    trainer.set_loss_function(loss)

    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, default="config/config.yaml", help="Config path")
    parser.add_argument("--ckpt", type=str, default=None)
    args = parser.parse_args()
    train(args)
