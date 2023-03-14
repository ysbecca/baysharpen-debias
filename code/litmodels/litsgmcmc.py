import torch
from torchvision.utils import make_grid
from torch.optim import Adam, SGD, lr_scheduler
import pytorch_lightning as pl
import numpy as np
import os
import torch.nn as nn

import torchvision.models as models
from pytorch_lightning.callbacks import LearningRateMonitor

from nnmodels.resnet import *
from nnmodels.vwresnet import *
from nnmodels.alexnet import *

import torch.nn.functional as F

from global_config import *


class LitSGMCMC(pl.LightningModule):
    def __init__(self,
            args,
            num_batches,
            iterations,
            model_desc,
            datasize,
            data_module
        ):
        super().__init__()

        if args.alpha < 1.0:
            self.is_hmc = True
        else:
            self.is_hmc = False

        # turn off optimizer
        if self.is_hmc:
            self.automatic_optimization = False

        self.args                       = args
        self.model_desc                 = model_desc
        self.num_batches                = num_batches
        self.iterations                 = iterations
        self.model                      = self.__get_model()

        self.moment                     = args.moments
        self.best_model_epoch           = 0
        self.best_val_loss              = 1000000
        self.datasize                   = datasize
        self.ious                       = None
        self.weight_decay               = 1e-4
        self.save_hyperparameters()

        self.data_module                = data_module

        self.manual_device              = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if not self.moment and not os.path.isdir(f"{MOMENTS_DIR}{self.model_desc}"):
            print(f"[INFO] Creating a moments directory for: {self.model_desc}")
            os.mkdir(f"{MOMENTS_DIR}{self.model_desc}")
            # set epoch = 0 since first train
            np.save(f"{MOMENTS_DIR}{self.model_desc}/last_epoch.npy", np.array([0]))

        # hack since we cannot manually set self.current_epoch
        if self.args.ckpt:
            # check the last epoch
            try:
                with open(f"{MOMENTS_DIR}{self.model_desc}/last_epoch.npy", 'rb') as f:
                    self.start_epoch = int(np.load(f)[0])
            except:
                self.start_epoch = 0

            if self.start_epoch:
                path = f"{MOMENTS_DIR}{self.args.model_desc}/{self.args.ckpt}.pt"
                self.__load_state_dict(path)

                print(f"[INFO] Loaded checkpoint at {self.args.ckpt}.pt")
                self.model.train()

            print(f"[INFO] Current epoch: {(self.current_epoch + self.start_epoch)}")

        else:
            self.start_epoch = 0


    def set_ious(self, ious):
        self.ious = ious

    def __get_model(self):

        if IS_LOCAL:
            model = alexnet(num_classes=NUM_CLASSES[self.args.dataset_code])

        else:

            if self.args.model_code == RESNET6_CODE:
                model = resnet6vw(num_classes=NUM_CLASSES[self.args.dataset_code])

            elif self.args.model_code == RESNET10_CODE:
                if self.args.dataset_code == COCO_PLACES_CODE and (not self.args.use_vw_flag_coco):
                    model = ResNet10(self.args.dataset_code, num_classes=NUM_CLASSES[self.args.dataset_code])
                else:
                    model = resnet10vw(num_classes=NUM_CLASSES[self.args.dataset_code])

            elif self.args.model_code == RESNET14_CODE:
                if self.args.dataset_code == COCO_PLACES_CODE and (not self.args.use_vw_flag_coco):
                    model = ResNet14(self.args.dataset_code, num_classes=NUM_CLASSES[self.args.dataset_code])
                else:
                    model = resnet14vw(num_classes=NUM_CLASSES[self.args.dataset_code])

            elif self.args.model_code == RESNET_CODE:
            
                if self.args.dataset_code == COCO_PLACES_CODE and (not self.args.use_vw_flag_coco):
                    model = ResNet18(self.args.dataset_code, num_classes=NUM_CLASSES[self.args.dataset_code])
                else:

                    model = nresnet18(
                        num_classes=NUM_CLASSES[self.args.dataset_code]
                    )

                    if self.args.pretrained:
                        # load checkpoint for pre-trained BAR on ImageNet100
                        ckpt = torch.load(CKPTS[self.args.dataset_code])
                        weights = ckpt['state_dict']
                        new_weights = dict()

                        for key, val in weights.items():
                            new_weights['.'.join(key.split('.')[1:])] = val

                        if self.args.dataset_code == BAR_CODE:
                            new_weights['fc.weight'] = torch.nn.init.xavier_uniform_(
                                torch.rand((6, 512)))
                            new_weights['fc.bias'] = torch.Tensor((6))

                        model.load_state_dict(new_weights)

            elif self.args.model_code == RESNET34_CODE:
                model = resnet34vw(num_classes=NUM_CLASSES[self.args.dataset_code])

            elif self.args.model_code == RESNET50_CODE:
                model = resnet50vw(num_classes=NUM_CLASSES[self.args.dataset_code])

            elif self.args.model_code == ALEX_CODE:
                model = alexnet(num_classes=NUM_CLASSES[self.args.dataset_code])

            else:
                model = None
                print("[ERR]: invalid model code", self.args.model_code)

        return model

    def __load_state_dict(self, path, iou_threshold=0.0):
        """ for both checkpoint and moment """

        state_dict = torch.load(path, map_location=self.manual_device)
        self.model.load_state_dict(state_dict)

    def weighting_function(self, epistemics):
        return torch.pow((1.0 + epistemics), self.args.kappa)

    def __clear_classifier_params(self, partial_model):
        """ Set random weights and biases for given layers """
        if type(partial_model) == torch.nn.Sequential:
            layers = list(partial_model)
        else:
            layers = [partial_model]

        for i in range(len(layers)):
            m = layers[i]
            if type(m) == nn.Linear or type(m) == nn.Conv2d:
                m.weight.data = torch.randn(m.weight.size())*.01
                m.bias.data = torch.zeros(m.bias.size())
                m.weight.requires_grad = True
                m.bias.requires_grad = True

        return partial_model

    def configure_optimizers(self):

        # SG-HMC
        if self.is_hmc:
            return None
        else:
            # SG-LD
            if self.args.optim_code == 'adam':
                return Adam(
                    params=filter(lambda p: p.requires_grad, self.parameters()),
                    lr=self.args.lr,
                    weight_decay=5e-3
                )
            elif self.args.optim_code == 'sgd':
                return SGD(
                    self.parameters(),
                    lr=self.args.lr,
                    momentum=1.0 - self.args.alpha,
                    weight_decay=self.weight_decay,
                )

    def standard_loss(self, y_pred, y):

        return F.nll_loss(y_pred, y.long(), weight=torch.Tensor(CLASS_WEIGHTS[self.args.dataset_code]).to(self.device))

    def accuracy(self, y_pred, y):
        acc = (y == torch.argmax(y_pred, dim=1)).sum() / torch.tensor(y.shape[0])
        return acc.item() * 100.0

    def noise_loss(self, lr):
        noise_loss = 0.0
        noise_std = (2 / lr * self.args.alpha)**0.5
        for var in self.model.parameters():
            means = torch.zeros(var.size()).to(self.device)
            noise_loss += torch.sum(var * torch.normal(means, std=noise_std).to(self.device))

        return noise_loss

    def update_params(self, lr):
        for p in self.model.parameters():
            if not hasattr(p,'buf'):
                p.buf = torch.zeros(p.size()).to(self.device)
            d_p = p.grad.data
            d_p.add_(p.data, alpha=self.weight_decay)

            buf_new = (1 - self.args.alpha) * p.buf - lr * d_p
            if ((self.current_epoch + self.start_epoch) % self.args.cycle_length) + 1 > (self.args.cycle_length - self.args.models_per_cycle):
                eps = torch.randn(p.size()).to(self.device)
                buf_new += (2.0 * self.args.lr * self.args.alpha * self.args.temperature / self.datasize)**.5 * eps
            p.data.add_(buf_new)
            p.buf = buf_new

    def adjust_learning_rate(self, batch_idx):
        rcounter = (self.current_epoch + self.start_epoch) * self.num_batches + batch_idx

        cos_inner = np.pi * (rcounter % (self.iterations // self.args.cycles))
        cos_inner /= self.iterations // self.args.cycles
        cos_out = np.cos(cos_inner) + 1
        lr = 0.5 * cos_out * self.args.lr

        if not self.is_hmc:
            # the cSG-HMC code does not change the lr here, bc does custom param update
            for param_group in self.optimizers().param_groups:
                param_group['lr'] = lr

        return lr

    def forward(self, x):
        outputs = self.model(x)

        return outputs

    def training_step(self, batch, batch_idx):
        self.model.zero_grad()

        # ======= NORMAL STUFF: FWD BCKWD ===============================
        x, y, indices = batch
        logits = self.forward(x)

        y_pred = F.softmax(logits, dim=1)
        y = torch.squeeze(y)

        # ======= Epistemic uncertainties ==========================================
        epistemics = []
        weights = None

        if self.args.epiwt and ((self.current_epoch + self.start_epoch) % self.args.cycle_length + 1 == self.args.cycle_length):
            # which index in cycle is this moment
            moment_id = self.moment % self.args.models_per_cycle

            # save in correct indices
            self.data_module.train_set.p_hats[moment_id][indices.cpu()] = y_pred.detach().cpu().numpy()

        # if not first cycle, use epis for dynamic upweighting
        if self.args.epiwt and ((self.current_epoch + self.start_epoch) > self.args.cycle_length):
            # p_bars = y pred, p_hats = p_theta_i(y | x)
            p_hats = self.data_module.train_set.p_hats[:, indices.cpu()]
            p_bars = np.mean(p_hats, axis=0)

            # to use full sample from posterior, use self.moment instead of MODELS_PER_CYCLE
            temp = (p_hats - np.tile(np.expand_dims(p_bars, 0), [self.args.models_per_cycle, 1, 1]))**2
            # 1 for testing, or self.moment for cumulative across cycles
            epistemics = np.sqrt(np.sum(temp, axis=0)) / self.args.models_per_cycle

            epistemics = torch.Tensor([e[i] for e, i in zip(epistemics, y)]).double().to(self.device)
            weights = self.weighting_function(epistemics)

        # ======= LR ADJUST AND LOSS =====================================
        lr = self.adjust_learning_rate(batch_idx)

        if not self.is_hmc:
            if ((self.current_epoch + self.start_epoch) % self.args.cycle_length) + 1 > (self.args.cycle_length - self.args.models_per_cycle):
                loss_noise = self.noise_loss(lr) * (self.args.temperature / self.datasize)**.5

                if weights:
                    # use weighted loss with noise
                    loss = self.standard_loss(y_pred, y, weights=weights) + loss_noise
                loss = self.standard_loss(y_pred, y) + loss_noise
            else:
                loss = self.standard_loss(y_pred, y)
        else:
            # cSG-HMC regular loss
            loss = self.standard_loss(y_pred, y)
            self.manual_backward(loss)

            # this is the update
            self.update_params(lr)

        # ======= ACCURACY =================================================
        acc = self.accuracy(y_pred, y)


        # ======= LOGGING  =================================================
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('lr', lr, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return {'loss': loss}

    def predict_custom(self):
        """ full predict and epis calculations on all loaders defined in self.loaders """

        preds, epis, accs = {}, {}, {}
        for key, loader in self.loaders.items():
            m_logits = {i: [] for i in range(self.moment)}
            for m in range(self.moment):
                moment_path = f"{MOMENTS_DIR}{self.args.model_desc}/moment_{m}.pt"                
                self.__load_state_dict(moment_path)

                if self.args.num_gpus > 1:
                    self.model = nn.DataParallel(self.model)

                self.model = self.model.to(self.manual_device)
                self.model.eval()

                with torch.no_grad():
                    for batch in loader:
                        x, _, _ = batch
                        x = x.to(self.manual_device)
                        m_logits[m].append(F.softmax(self.model(x), dim=1))

                        if self.args.dev_run:
                            break

            # [N_MOMENTS, N_SAMPLES, N_CLASSES]
            m_logits = torch.stack([torch.cat(m_logits[m], dim=0) for i in range(self.moment)])

            # key: [SAMPLES, N_CLASSES]
            preds[key] = m_logits.mean(dim=0)
            cut = len(preds[key])

            targets = torch.Tensor(self.targets[key][:cut]).to(self.manual_device)

            # accumulate epistemic uncertainties
            temp = (m_logits - preds[key].expand((self.moment, *preds[key].shape)))**2
            epis_ = torch.sqrt(torch.sum(temp, axis=0)) / self.moment

            epis_ = torch.Tensor([e[int(i)] for e, i in zip(epis_, targets)]).double().to(self.manual_device)
            epis[key] = epis_ * SCALAR_EPI

            correct = (targets == torch.argmax(preds[key], dim=1))

            acc = self.accuracy(preds[key], targets)
            accs[key] = acc

        return preds, epis, accs

    def set_targets(self, targets):
        self.targets = targets    

    def set_loaders(self, loaders):
        self.loaders = loaders
        print(f"[INFO] set loaders: {self.loaders.keys()}")

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch

        logits = self.forward(x)
        y_pred = F.softmax(logits, dim=1)
        y = torch.squeeze(y)


        loss = self.standard_loss(y_pred, y)
        acc = self.accuracy(y_pred, y)

        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        print(batch_idx, end=" ")
        return {'val_loss': loss}

    def validation_epoch_end(self, val_step_outputs):
        # compute val loss over epoch
        val_loss_epoch = torch.Tensor([x['val_loss'] for x in val_step_outputs]).sum().item()

        if val_loss_epoch < self.best_val_loss:
            self.best_model_epoch = self.current_epoch + self.start_epoch
            self.best_val_loss = val_loss_epoch

        # print(f"[INFO] current epoch {self.current_epoch}, start_epoch {self.start_epoch}, cycle length {self.args.cycle_length}")
        # print(f"LHS {((self.current_epoch + self.start_epoch) % self.args.cycle_length) + 1}, RHS {self.args.cycle_length - self.args.models_per_cycle}")
        if ((self.current_epoch + self.start_epoch) % self.args.cycle_length) + 1 > self.args.cycle_length - self.args.models_per_cycle:

            print(f"SAVING MOMENT {self.moment}, epoch no. {(self.current_epoch + self.start_epoch)}")
            torch.save(self.model.state_dict(), f"{MOMENTS_DIR}{self.model_desc}/moment_{self.moment}.pt")

            self.moment += 1

        # checkpoint every DEFAULT_CKPT_FREQ epochs for long cycles in case get kicked off
        elif (self.current_epoch % DEFAULT_CKPT_FREQ == 0) and not self.args.dev_run:

            with open(f"{MOMENTS_DIR}{self.model_desc}/last_epoch.npy", 'wb') as f:
                np.save(f, np.array([self.current_epoch + self.start_epoch]))

            if not self.args.dev_run and self.args.dataset_code == IMAGENET_CODE:
                print(f"SAVING CKPT, epoch no. {(self.current_epoch + self.start_epoch)}")
                torch.save(self.model.state_dict(), f"{MOMENTS_DIR}{self.model_desc}/last.pt")
                torch.save(self.model.state_dict(), f"{MOMENTS_DIR}{self.model_desc}/{(self.current_epoch + self.start_epoch)}.pt")
