import torch
from torchvision.utils import make_grid
from torch.optim import Adam, SGD, lr_scheduler
import pytorch_lightning as pl
import numpy as np
import os
import torch.nn as nn

import torchvision.models as models
from pytorch_lightning.callbacks import LearningRateMonitor
from torch.optim.lr_scheduler import StepLR, MultiStepLR

from nnmodels.resnet import *
from nnmodels.vwresnet import *

# from nnmodels.alexnet import *

from torchvision.models import alexnet

import torch.nn.functional as F

from global_config import *



class LitDetCNN(pl.LightningModule):
    def __init__(self, args, model_desc):
        super().__init__()


        self.args                       = args
        self.model_desc                 = model_desc

        self.model                      = self.__get_model()

        self.best_model_epoch           = 0
        self.best_val_loss              = 1000000
        # self.ious                       = None
        self.weight_decay               = 1e-4
        self.save_hyperparameters()

        self.configure_optimizers()

        if not os.path.isdir(f"{MOMENTS_DIR}{self.model_desc}"):
            print(f"[INFO] Creating a checkpoints directory for: {self.model_desc}")
            os.mkdir(f"{MOMENTS_DIR}{self.model_desc}")
            # set epoch = 0 since first train
            np.save(f"{MOMENTS_DIR}{self.model_desc}/best_epoch.npy", np.array([0]))


        if self.args.pretrained:
            self.load_state_dict()

        # hack since we cannot manually set self.current_epoch
        if self.args.ckpt:
            try:
                with open(f"{MOMENTS_DIR}{self.model_desc}/best_epoch.npy", 'rb') as f:
                    self.start_epoch = int(np.load(f)[0])
            except:
                self.start_epoch = 0

            if self.start_epoch:
                self.load_state_dict()

                print(f"[INFO] Loaded checkpoint at {self.args.ckpt}.pt")
                self.model.train()

            print(f"[INFO] Current epoch: {(self.current_epoch + self.start_epoch)}")
        else:
            self.start_epoch = 0


    def imnet_accuracy(self, output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res


    def __get_model(self):

        if IS_LOCAL:
            model = alexnet(num_classes=NUM_CLASSES[self.args.dataset_code])
        else:
            if self.args.model_code == RESNET_CODE:
                # model = ResNet18(self.args.dataset_code, num_classes=NUM_CLASSES[self.args.dataset_code])
                model = nresnet18(num_classes=NUM_CLASSES[self.args.dataset_code])
            elif self.args.model_code == RESNET50_CODE:
                model = ResNet50(num_classes=NUM_CLASSES[self.args.dataset_code])
            elif self.args.model_code == ALEX_CODE:
                model = alexnet(num_classes=NUM_CLASSES[self.args.dataset_code])
            else:
                model = None
                print("[ERR]: invalid model code", self.args.model_code)

        return model


    def load_state_dict(self):
        """ filename of the model checkpoint and moment """

        ckpt = self.args.ckpt if self.args.ckpt else "best"

        if self.args.pretrained:
            ckpt = torch.load(CKPTS[self.args.dataset_code])
            print(f"[INFO] Loading checkpoint {CKPTS[self.args.dataset_code]}")
            weights = ckpt['state_dict']
            new_weights = dict()

            for key, val in weights.items():
                new_weights['.'.join(key.split('.')[1:])] = val

            self.model.load_state_dict(new_weights)

        else:

            path = f"{MOMENTS_DIR}{self.args.model_desc}/{ckpt}.pt"
            print(f"[INFO] Loading checkpoint from {ckpt}.pt")
            state_dict = torch.load(path, map_location=self.device)

            self.model.load_state_dict(state_dict['model_state_dict'])
            self.optim.load_state_dict(state_dict['optim_state_dict'])
            self.scheduler.load_state_dict(state_dict['sched_state_dict'])


    def configure_optimizers(self):

        if self.args.optim_code == 'adam':
            optim = Adam(
                self.parameters(),
                lr=self.args.lr,
                weight_decay=5e-3
            )
        elif self.args.optim_code == 'sgd':
            optim = SGD(
                self.parameters(),
                lr=self.args.lr,
                momentum=0.9,
                weight_decay=1e-4,
            )

        self.optim = optim

        self.scheduler = MultiStepLR(
            optim,
            milestones=DECAY_MILESTONES[self.args.dataset_code],
            gamma=0.1
        )
        return [self.optim], [{"scheduler": self.scheduler}]


    def standard_loss(self, y_pred, y):
        return F.cross_entropy(y_pred, y)


    def accuracy(self, y_pred, y):
        acc = (y[:len(y_pred)] == torch.argmax(y_pred, dim=1)).sum() / len(y_pred)
        return acc.item() * 100.0

    def forward(self, x):
        if isinstance(x, list):
            x_, _, _ = x
        else:
            x_ = x

        outputs = self.model(x_)

        return outputs

    def predict_step(self, batch, batch_idx):

        x, _, _ = batch
        logits = self.forward(x)
        pred = F.softmax(logits, dim=1)

        return pred


    def training_step(self, batch, batch_idx):

        x, y, _ = batch
        logits = self.forward(x)

        y_pred = F.softmax(logits, dim=1)

        y = torch.squeeze(y)

        loss = self.standard_loss(y_pred, y)
        acc = self.accuracy(y_pred, y)

        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)


        return {'loss': loss}


    def validation_step(self, batch, batch_idx):
        x, y, _ = batch

        logits = self.forward(x)
        y_pred = F.softmax(logits, dim=1)

        y = torch.squeeze(y)

        loss = self.standard_loss(y_pred, y)
        if self.args.dataset_code == IMAGENET_CODE:
            acc1, acc5 = self.imnet_accuracy(y_pred, y, topk=(1, 5))
            self.log('val_acc1', acc1, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log('val_acc5', acc5, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        else:
            acc = self.accuracy(y_pred, y)
            self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        #self.log('lr', self.scheduler.get_last_lr(), on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {'val_loss': loss}


    def validation_epoch_end(self, val_step_outputs):
        # compute val loss over epoch
        val_loss_epoch = torch.Tensor([x['val_loss'] for x in val_step_outputs]).sum().item()

        # checkpoint every DEFAULT_CKPT_FREQ epochs for long cycles in case get kicked off
        # if (self.current_epoch % DEFAULT_CKPT_FREQ == 0) and not self.args.dev_run:
        # ^ No need to checkpoint because on desktop GPU now.
        if not self.args.dev_run:

            ckpt = {
                'model_state_dict': self.model.state_dict(),
                'optim_state_dict': self.optim.state_dict(),
                'sched_state_dict': self.scheduler.state_dict(),
            }

            if val_loss_epoch <= self.best_val_loss:
                self.best_model_epoch = self.current_epoch + self.start_epoch
                self.best_val_loss = val_loss_epoch
                print(f"[SAVING] best loss, epoch no. {(self.current_epoch + self.start_epoch)}")

                with open(f"{MOMENTS_DIR}{self.model_desc}/best_epoch.npy", 'wb') as f:
                    np.save(f, np.array([self.current_epoch + self.start_epoch]))

                torch.save(ckpt, f"{MOMENTS_DIR}{self.model_desc}/best.pt")

            # torch.save(ckpt, f"{MOMENTS_DIR}{self.model_desc}/last.pt")
            # torch.save(ckpt, f"{MOMENTS_DIR}{self.model_desc}/{(self.current_epoch + self.start_epoch)}.pt")
