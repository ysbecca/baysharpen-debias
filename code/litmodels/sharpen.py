import torch
from torchvision.utils import make_grid
from torch.optim import Adam, SGD, lr_scheduler
import pytorch_lightning as pl
import numpy as np
from torchvision import transforms
from torchvision.utils import make_grid
import os
import torch.nn as nn

from PIL import Image

import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter
from pytorch_lightning.callbacks import LearningRateMonitor

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from pytorch_pcgrad.pcgrad import PCGrad

from nnmodels.resnet import *
from nnmodels.alexnet import *
from nnmodels.vwresnet import *


import torch.nn.functional as F

from global_config import *


class Sharpen():
    def __init__(self, args, model_desc, data_module):

        self.manual_device              = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # torch.cuda.set_device(1)
        print(f"[INFO] Manual device detected, ", self.manual_device)

        self.args                       = args
        self.model_desc                 = model_desc
        self.models                     = self.__get_models()

        print(f"[INFO] Set up {len(self.models)} models...")

        self.moment                     = self.args.moments
        self.best_model_epoch           = 0
        self.best_val_loss              = 1000000
        self.weight_decay               = 1e-4
        # self.save_hyperparameters()

        self.cams                       = self.__prep_cams()
        self.current_batch_means        = None

        self.optimizers                 = self.get_optimizers()

        self.data_module                = data_module
        self.dataset                    = data_module.valid_set

        self.train_key                  = "train"


        print(f"[INFO] Set up {len(self.optimizers)} optimizers...")

        if not os.path.isdir(f"{MOMENTS_DIR}{self.model_desc}"):
            print(f"[INFO] Creating a moments directory for: {self.model_desc}")
            os.mkdir(f"{MOMENTS_DIR}{self.model_desc}")

        # for cam
        self.data_transform             = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

    def __get_models(self):
        assert self.args.model_code == RESNET_CODE

        models = []
        print(f"[INFO] Total moments to load: {self.args.moments}")
        for m in range(self.args.moments):

            # load other models too.... TODO 
            
            if self.args.dataset_code == COCO_PLACES_CODE and (not self.args.use_vw_flag_coco):
                model = ResNet18(
                    self.args.dataset_code,
                    num_classes=NUM_CLASSES[self.args.dataset_code]
                )
            else:
                model = nresnet18(num_classes=NUM_CLASSES[self.args.dataset_code])
            if not IS_LOCAL:
                path = f"{MOMENTS_DIR}{self.args.moment_desc}/moment_{m}.pt"
                model = self.__load_state_dict(model, path)
            models.append(model)

        return models

    def __prep_cams(self):
        cams = []

        for i in range(len(self.models)):
            # hook every model
            c = GradCAM(
                model=self.models[i],
                target_layers=[self.models[i].layer4[1].conv2],
                use_cuda=(self.manual_device != "cpu"),
            )
            cams.append(c)

        return cams

    def __load_state_dict(self, model, path):
        """ for both checkpoint and moment """

        state_dict = torch.load(path, map_location=self.manual_device)
        model.load_state_dict(state_dict)

        return self.__freeze_model(model)

    def __freeze_model(self, model):
        # with freezing for sharpening
        for param in model.parameters():
            param.requires_grad = False

        if self.args.dataset_code == COCO_PLACES_CODE:
            for param in model.linear.parameters():
                param.requires_grad = True
        else:
            for param in model.fc.parameters():
                param.requires_grad = True

        model.train()
        return model

    def __unfreeze_model(self, model):
        # unfreeze all layers
        for param in model.parameters():
            param.requires_grad = True

        return model

    def get_optimizers(self):
        assert self.args.optim_code == 'sgd'

        optims = []
        for i in range(self.args.moments):
            optims.append(SGD(
                self.models[i].parameters(),
                lr=self.args.lr,
                momentum=0.9,
                weight_decay=self.weight_decay,
            ))

        if self.args.loss_type == PCGRAD_LOSS:
            optims = [PCGrad(o) for o in optims]

        return optims

    def standard_loss(self, y_pred, y, weights=[]):
        if len(weights):
            loss = F.nll_loss(y_pred, y.long(), reduction="none")
            loss *= weights

            return loss.sum() / len(y)
        else:
            return F.nll_loss(y_pred, y.long())

    def weighting_function(self, epistemics):
        return torch.pow((1.0 + epistemics), self.args.kappa)

    def accuracy(self, y_pred, y):
        acc = (y == torch.argmax(y_pred, dim=1)).sum() / torch.tensor(y.shape[0])
        return acc.item() * 100.0

    def add_border(self, vis, border_size):
        size = vis.shape[0]

        vis[0:size, 0:border_size, 0:border_size] = (255, 0, 0)
        vis[0:border_size, 0:size, 0:border_size] = (255, 0, 0)
        vis[0:size, (size - border_size):size, ] = (255, 0, 0)
        vis[(size - border_size):size,0:size, ] = (255, 0, 0)

        return vis

    def generate_visualisation(self, keys, epis, preds):
        """ generate grad-cam visualisation - sets defined by keys list """

        cams = {}
        if self.args.dev_run:
            cut = self.args.batch_size if self.args.batch_size < CAMS_TO_SHOW else CAMS_TO_SHOW
        else:
            cut = CAMS_TO_SHOW

        # unfreeze all models
        self.models = [self.__unfreeze_model(m) for m in self.models]
        for key in keys:
            key_cams = []
            # sort the epis to select which images to show
            weights = self.weighting_function(epis[key])
            # get indices of top N biggest weights (ie, highest uncertainties)
            select = torch.argsort(weights, descending=True).cpu().numpy()[:cut]

            x_select = self.data_module.get_select(key, select)
            y_select = self.targets[key][select]

            moment_cams = np.array([
                self.cams[j](
                    input_tensor=x_select,
                    targets=[ClassifierOutputTarget(t) for t in y_select]) 
                for j in range(self.args.moments)
            ])

            if self.args.dataset_code == BAR_CODE:
                x_select = self.data_module.get_select(key, select, vis=True)

            # [MOMENTS, CAMS_TO_SHOW, 224, 224]
            # compute mean across posterior
            mean_cams = np.mean(moment_cams, axis=0)

            # for red border on wrong images
            preds_select = np.argmax(preds[key][select].cpu().numpy(), axis=1)
            correct = (y_select == preds_select)

            size = mean_cams[0].shape[0]
            for i in range(cut):
                x = np.float32(x_select[i])
                x = np.moveaxis(x, 0, 2)
                if x.min() < 0 and x.max() > 1.0:
                    x = (x + x.min()) / x.max()
                c = show_cam_on_image(
                    x,
                    np.array(mean_cams[i]),
                    use_rgb=True
                )

                if not correct[i]:
                    c = self.add_border(c, 5)
                key_cams.append(torch.Tensor(c).type(dtype=torch.uint8))

            cams[key] = torch.stack(key_cams)
        self.models = [self.__freeze_model(m) for m in self.models]
        return cams

    def set_targets(self, targets):
        self.targets = targets    

    def set_loaders(self, loaders):
        self.loaders = loaders
        print(f"[INFO] set loaders: {self.loaders.keys()}")

    def train_sharpen(self, epochs=1):

        # manual logging
        tb_writer = SummaryWriter(f"{TB_LOGS_PATH}{self.model_desc}")

        print(f"[INFO] Set up logger...")

        # always use initial epis to select images for cams so we can track progression.
        init_epis = None

        for e in range(epochs):
            print(f"[INFO] Sharpening epoch {e}")

            mean_preds, epis, accs = self.predict_custom()
            if e == 0:
                init_epis = epis
            # use biased distribution to compute epis
            weights = self.weighting_function(epis[self.train_key])

            self.log_epis_accs(tb_writer, epis, accs, epoch=e)

            # which of these are important to see?
            keys_to_vis = list(self.loaders.keys() - 'train')
            cams = self.generate_visualisation(keys_to_vis, init_epis, mean_preds)

            # display these images on the board under their epoch number
            for key in self.loaders.keys():
                tb_writer.add_images(f"{key}", cams[key], global_step=e, dataformats="NHWC")

            for model_idx, m in enumerate(self.models):
                if self.args.num_gpus > 1:
                    m = nn.DataParallel(m)

                m = m.to(self.manual_device)
                m.train()

                i = 0
                m_preds = []

                accum_sharpen_loss, accum_std_loss = 0.0, 0.0
                for batch_idx, batch in enumerate(self.loaders[self.train_key]):
                    x, y, sample_idxes = batch

                    batch_size = len(x)
                    targets_batch = mean_preds[self.train_key][sample_idxes]
                    weights_batch = weights[sample_idxes]
                    i += batch_size

                    x = x.to(self.manual_device)
                    y = y.to(self.manual_device)
                    # compute loss / gradients using weights based on means as target
                    preds_batch = F.softmax(m(x), dim=1)
                    self.optimizers[model_idx].zero_grad()

                    sharpen_loss = self.standard_loss(preds_batch, torch.argmax(targets_batch, dim=1), weights=weights_batch)
                    std_loss = self.standard_loss(preds_batch, y, weights=weights_batch)

                    # loss types
                    if self.args.loss_type == WEIGHTED_LOSS:
                        loss = std_loss + (5 * sharpen_loss)
                    if self.args.loss_type == INTERLEAVING_LOSS and e % 2:
                        loss = std_loss
                    else:
                        loss = sharpen_loss

                    accum_sharpen_loss += sharpen_loss
                    accum_std_loss += std_loss

                    m_preds.append(preds_batch)

                    if self.args.loss_type == PCGRAD_LOSS:
                        self.optimizers[model_idx].pc_backward([sharpen_loss, std_loss])
                    else:
                        loss.backward()

                    self.optimizers[model_idx].step()

                    if self.args.dev_run:
                        break


                tb_writer.add_scalar(f"loss/m{model_idx}_sharpen_loss", accum_std_loss / len(weights), e)
                tb_writer.add_scalar(f"loss/m{model_idx}_std_loss", accum_sharpen_loss / len(weights), e)


        for i, m in enumerate(self.models):
            torch.save(m.state_dict(), f"{MOMENTS_DIR}{self.model_desc}/moment_{i}.pt")

    def log_epis_accs(self, tb_writer, epis, accs, epoch):
        """ this logs all dataset epis, not validation set epis. """
        for key in self.loaders.keys():
            tb_writer.add_scalar(f'epis/{key}_epis', epis[key].mean(), epoch)
            tb_writer.add_scalar(f'accs/{key}_acc', accs[key], epoch)

    def predict_custom(self):
        """ full predict and epis calculations on all loaders defined in self.loaders """

        self.loaders['train'] = self.data_module.train_dataloader(shuffle=False)

        preds, epis, accs = {}, {}, {}
        for key, loader in self.loaders.items():
            m_logits = {i: [] for i in range(len(self.models))}
            for m_idx, m in enumerate(self.models):
                if self.args.num_gpus > 1:
                    m = nn.DataParallel(m)

                m = m.to(self.manual_device)
                m.eval()
                with torch.no_grad():
                    for batch in loader:
                        x, _, _ = batch
                        x = x.to(self.manual_device)
                        m_logits[m_idx].append(F.softmax(m(x), dim=1))

                        if self.args.dev_run:
                            break

            # [N_MOMENTS, N_SAMPLES, N_CLASSES]
            m_logits = torch.stack([torch.cat(m_logits[i], dim=0) for i in range(len(self.models))])

            # key: [SAMPLES, N_CLASSES]
            preds[key] = m_logits.mean(dim=0)
            cut = len(preds[key])

            targets = torch.Tensor(self.targets[key][:cut]).to(self.manual_device)

            # accumulate epistemic uncertainties
            temp = (m_logits - preds[key].expand((self.args.moments, *preds[key].shape)))**2
            epis_ = torch.sqrt(torch.sum(temp, axis=0)) / self.args.moments

            epis_ = torch.Tensor([e[int(i)] for e, i in zip(epis_, targets)]).double().to(self.manual_device)
            epis[key] = epis_ * SCALAR_EPI

            correct = (targets == torch.argmax(preds[key], dim=1))

            acc = self.accuracy(preds[key], targets)
            accs[key] = acc

        self.loaders['train'] = self.data_module.train_dataloader(shuffle=True)
        return preds, epis, accs




