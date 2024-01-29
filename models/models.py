import torch
import torch.nn.functional as F

import pytorch_lightning as pl


class ClassificationNet(pl.LightningModule):
    def __init__(self, model, classes, in_channels=3, lr=1e-3):
        super().__init__()

        self.model = model
        self.in_channels = in_channels
        self.num_classes = classes
        self.lr = lr

    def forward(self, x):
        res = self.model(x)
        return res

    def cross_entropy_loss(self, pred, label):
        loss = F.cross_entropy(pred, label, label_smoothing=0.05)
        return loss

    def training_step(self, batch, batch_idx):
        img, label = batch

        pred = self.forward(img)
        loss = self.cross_entropy_loss(pred, label)

        pred_classes = torch.argmax(pred, dim=1)
        acc = torch.sum(pred_classes == label) / len(label)
        self.log('train/loss', loss)
        self.log('train/accuracy', acc)

        return loss

    def validation_step(self, batch, batch_idx):
        img, label = batch
        pred = self.forward(img)

        loss = self.cross_entropy_loss(pred, label)

        pred_classes = torch.argmax(pred, dim=1)
        acc = torch.sum(pred_classes == label) / len(label)
        self.log('val/loss', loss)
        self.log('val/accuracy', acc)
        return loss


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=1e-5)
        # optimizer = torch.optim.SGD(self.parameters(), lr=1e-2, momentum=0.9, weight_decay=0.0001)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.9, min_lr=1e-6, patience=2)
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
        # return optimizer
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,  # Changed scheduler to lr_scheduler
        }