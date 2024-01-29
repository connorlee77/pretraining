import argparse
import pytorch_lightning as pl
import torch
import wandb

from datasets.imagenet import ImageNetDataModule

from models.fast_scnn_512x640 import FastSCNN512x640
from models.models import ClassificationNet

from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy

pl.seed_everything(0, workers=True)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')

    parser.add_argument('--dataset', type=str, default='imagenet')
    parser.add_argument('--dataroot', type=str, default='/data/imagenet/imagenet')

    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=8,
                        help='Number of workers for dataloader')
    parser.add_argument('--num-epochs', type=int, default=500)

    parser.add_argument('--model', type=str)
    parser.add_argument('--learning-rate', type=float, default=1e-3)

    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--tag', default=None, type=str, help='Tag to append to wandb experiment name')

    opt = parser.parse_args()
    print(opt)

    nclasses = 0
    data_module = None
    if opt.dataset == 'imagenet':
        data_module = ImageNetDataModule(
            data_dir=opt.dataroot,
            batch_size=opt.batch_size,
            num_workers=opt.num_workers,
        )
        nclasses = 1000
        in_channels = 3
    else:
        raise ValueError('Unknown dataset {}'.format(opt.dataset))

    if opt.model == 'fastscnn':
        model = FastSCNN512x640(
            num_classes=nclasses,
            in_channels=in_channels,
            aux=False)
    else:
        raise ValueError('Unknown model {}'.format(opt.model))
        

    network = ClassificationNet(model=model, classes=nclasses, in_channels=in_channels, lr=opt.learning_rate)
    
    wandb_exp_name = "{}-{}".format(opt.model, opt.tag) if opt.tag else opt.model
    wandb_logger = WandbLogger(
        name=wandb_exp_name,
        project=opt.dataset,
        log_model=False)

    lr_monitor = LearningRateMonitor(logging_interval='step')

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath="{}/{}".format(opt.dataset, wandb_exp_name),
        monitor="val/loss",
        every_n_epochs=1,
        save_last=True,
        save_top_k=1,
        mode='min',
        auto_insert_metric_name=True
    )

    earlystop = EarlyStopping(
        monitor="val/loss",
        mode="min",
        patience=50,
        verbose=False)

    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer = pl.Trainer(
        profiler='simple',
        devices=[opt.device],
        accelerator='gpu',
        strategy='auto',
        max_epochs=opt.num_epochs,
        callbacks=[checkpoint_callback, lr_monitor, earlystop],
        log_every_n_steps=10,
        logger=wandb_logger,
        sync_batchnorm=True,
        # overfit_batches=0.01, # Use this for debugging
    )
    
    torch.set_float32_matmul_precision('medium')
    trainer.fit(network, datamodule=data_module)
    wandb.finish()
