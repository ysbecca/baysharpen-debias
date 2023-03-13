import yaml
import sys
import os
import numpy as np
import math
import csv
import pickle

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torchvision import transforms
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath('..'))

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import argparse
import torch
import random


from dataset_models.synbols_dm import *
from dataset_models.biased_mnist import *
from dataset_models.biased_mnist_dm import *
from dataset_models.coco_on_places import *
from dataset_models.coco_on_places_dm import *
from dataset_models.bar import *
from dataset_models.bar_dm import *


from litmodels.litsgmcmc import *
from litmodels.litdetcnn import *
from litmodels.sharpen import *

from utils import *
from global_config import *


# Set up training arguments and parse
parser = argparse.ArgumentParser(description='training network')

parser.add_argument(
    '--task_id', type=int, default=-1,
    help='task id; if present, use this to look for config yaml file that overrides all other args')
parser.add_argument(
    '--config_name', type=str, default=None,
    help='name of config file, if any')
parser.add_argument(
    '--dataset_code', type=str, default=IMAGENET_CODE,
    help='the dataset string code')
parser.add_argument(
    '--lit_code', type=str, default=CSGMCMC_CODE,
    help='lightning model code for training')
parser.add_argument(
    '--model_code', type=str, default=DEFAULT_MODEL,
    help='model code for model architecture to use')
parser.add_argument(
    '--model_desc', type=str, default="test",
    help='model description; if not defined, set as test')
parser.add_argument(
    '--ckpt', type=str, default=None,
    help='model checkpoint to run from')

parser.add_argument(
    '--optim_code', type=str, default=DEFAULT_OPTIM,
    help='optimizer code for training')

parser.add_argument(
    '--max_epochs', type=int, default=DEFAULT_MAX_EPOCH,
    help='max umber of epochs to train for')
parser.add_argument(
    '--batch_size', type=int, default=DEFAULT_BATCH_SIZE,
    help='training batch size')
parser.add_argument(
    '--momentum', type=float, default=DEFAULT_MOMENTUM,
    help='optimizer momentum')
parser.add_argument(
    '--lr', type=float, default=DEFAULT_LR,
    help='initial learning rate')
parser.add_argument(
    '--result_file', type=str, default=DEFAULT_RESULT_FILE,
    help='log results (append) to this csv')
parser.add_argument(
    '--cycle_length', type=int, default=DEFAULT_CYCLE_LENGTH,
    help='length of cycle by default')
parser.add_argument(
    '--num_gpus', type=int, default=GPU_COUNT,
    help='number of gpus')
parser.add_argument(
    '--dev_run', type=bool, default=False,
    help='runs a dev testing run on limited num of batches')

# hacky but using for sweeps temporarily
parser.add_argument(
    '--use_vw_flag_coco', type=bool, default=False,
    help='use vw resnet for coco')
parser.add_argument(
    '--bmnist_size', type=int, default=160,
    help='bmnist image size')

# temp for debugging imagenet
parser.add_argument(
    '--small_imagenet', type=bool, default=False,
    help='small imagenet for debugging')

parser.add_argument(
    '--pretrained', type=bool, default=False,
    help='start training from pretrained imagenet weights')
parser.add_argument(
    '--keep_fc', type=bool, default=False,
    help='keep fc layers but use loss noise immediately')
parser.add_argument(
    '--alpha', type=float, default=0.3,
    help='1: SGLD; <1: HMC')

parser.add_argument(
    '--epiwt', type=bool, default=False,
    help='epi loss upweighting for training')
parser.add_argument(
    '--kappa', type=float, default=5.0,
    help='loss weighting scalar')

# if testing on a pre-trained model, set --max_epochs 0 and manually set moments
parser.add_argument(
    '--moments', type=int, default=0,
    help='for testing only on pre-trained model; set manually')
parser.add_argument(
    '--temperature', type=float, default=1./IMAGENET_TRAINSIZE,
    help='temperature (default: 1/dataset_size)')
parser.add_argument(
    '--cycles', type=int, default=DEFAULT_CYCLES,
    help='number of stochastic cycles during training')
parser.add_argument(
    '--models_per_cycle', type=int, default=MODELS_PER_CYCLE,
    help='number of moments per cycle')


# spurious synbols dataset
parser.add_argument(
    '--bg_mode', type=str, default=GRAD_BG,
    help='for spurious synbols: background mode')
parser.add_argument(
    '--spurious_mode', type=str, default=SPURIOUS_MODE_REG,
    help='for spurious synbols: type of spurious bias')
parser.add_argument(
    '--spurious_ratio', type=float, default=DEFAULT_SPURIOUS_RATIO,
    help='for spurious synbols: ratio for class samples w spurious feature')
parser.add_argument(
    '--sf_size', type=int, default=MASK_SIZE,
    help='size of regional spurious feature mask size')


parser.add_argument(
    '--iou_overlap', type=float, default=0.0,
    help='overlap threshold for unit cancellation')
parser.add_argument(
    '--loss_type', type=str, default=SHARPEN_LOSS,
    help='sharpened loss type')
parser.add_argument(
    '--loss_scalar', type=int, default=SHARPEN_LOSS_SCALAR,
    help='size of regional spurious feature mask size')
parser.add_argument(
    '--moment_desc', type=str, default="shp-synbols-test",
    help='model description from which to read the moment checkpoints')

args = parser.parse_args()

if args.task_id > -1:
    # override with yaml configs if necessary
    args = override_from_config(args)




print("=======================================")
print("         TRAINING PARAMS               ")
print("=======================================")
for arg in vars(args):
    print(F"{arg:>20} {getattr(args, arg)}")


# seeds
torch.manual_seed(DEFAULT_SEED)
np.random.seed(DEFAULT_SEED)
random.seed(DEFAULT_SEED)



########################################################################
#
#           Logger
#
########################################################################

if len(args.model_desc):
    model_desc = args.model_desc
else:
    model_desc = get_model_desc(args, include_task_id=True)

print(f"MODEL DESCRIPTION: {model_desc:>10}")


logger = TensorBoardLogger(TB_LOGS_PATH, name=model_desc)



########################################################################
#
#           Dataset
#
########################################################################

batch_size = args.batch_size*args.num_gpus if args.num_gpus else args.batch_size

# TODO replace this by DataModule Factory
if args.dataset_code == IMAGENET_CODE:
    data_module = DataModuleImageNet(
        batch_size=batch_size,
        small_imagenet=args.small_imagenet,
    )
elif args.dataset_code == SYNBOLS_CODE:
    data_module = DataModuleSpuriousSynbols(
        batch_size=batch_size,
        args=args,
        seg_mode=False,
    )
elif args.dataset_code == BMNIST_CODE:
    data_module = DataModuleBiasedMNIST(
        batch_size=batch_size,
        args=args,
    )
elif args.dataset_code == COCO_PLACES_CODE:
    data_module = DataModuleCocoPlaces(
        batch_size=batch_size,
        args=args,
    )
elif args.dataset_code == IMAGENET100_CODE:
    data_module = DataModuleImageNet(
        batch_size=batch_size,
        small_imagenet=args.small_imagenet,
        subset100=True,
    )
elif args.dataset_code == BAR_CODE:
    data_module = DataModuleBAR(
        batch_size=batch_size,
        args=args,
    )
else:
    print("[ERR]: no recognised data code", args.dataset_code)
    exit()


# force setup so we can compute number of batches
data_module.setup()
num_batches = len(data_module.train_set) / batch_size + 1
iterations = num_batches * args.max_epochs



########################################################################
#
#           Setup Lightning module
#
########################################################################

def fetch_loaders_and_targets(data_module):
    loaders, targets = {}, {}

    for key in data_module.keys:
        targets[key] = data_module.get_targets(key)
        loaders[key] = data_module.get_dataloader(key=key)

    return loaders, targets

loaders, targets = fetch_loaders_and_targets(data_module)

if args.lit_code == CSGMCMC_CODE:
    model = LitSGMCMC(
        args,
        num_batches=num_batches,
        iterations=iterations,
        model_desc=model_desc,
        datasize=IMAGENET_TRAINSIZE,
        data_module=data_module,
    )

    model.set_loaders(loaders)
    model.set_targets(targets)

elif args.lit_code == DET_CODE:
    model = LitDetCNN(
        args,
        model_desc=model_desc,
    )

elif args.lit_code == SHARPEN_CODE:
    model = Sharpen(
        args,
        model_desc=model_desc,
        data_module=data_module,
    )
else:
    print("[ERR]: lit code not recognised", args.lit_code)
    exit()


########################################################################
#
#           Training
#
########################################################################

val_steps = 1 if args.dev_run else -1
pred_batches = 1 if args.dev_run else 1.0

if args.num_gpus > 1:
    accelerator = "ddp"
else:
    accelerator = None


if args.lit_code == DET_CODE:
    callbacks = []
    # callbacks = [EarlyStopping(monitor="val_loss", mode="min", patience=3, verbose=True)]
else:
    callbacks = []

# init trainer
trainer = pl.Trainer(
	gpus=args.num_gpus,
    accelerator=accelerator,
	max_epochs=args.max_epochs,
    logger=logger,
    callbacks=callbacks,
    fast_dev_run=args.dev_run,
    num_sanity_val_steps=val_steps, # run complete pass through val set
    limit_predict_batches=pred_batches,
)


if args.max_epochs and not (args.lit_code == SHARPEN_CODE):

    trainer.fit(model, data_module)
elif (args.lit_code == SHARPEN_CODE):

    model.set_loaders(loaders)
    model.set_targets(targets)

    model.train_sharpen(
        epochs=args.max_epochs,
    )

########################################################################
#
#           Inference and evaluation
#
########################################################################

def save_outputs(preds, epis, extra=""):
    # to avoid overriding stuff accidentally
    char = "_dev" if args.dev_run else ""

    # save model preds and uncertainties
    np.save(f"{MOMENTS_DIR}{model_desc}/preds{char}{extra}.npy", np.argmax(preds.detach().cpu(), axis=1))
    np.save(f"{MOMENTS_DIR}{model_desc}/epis{char}{extra}.npy", np.array(epis.detach().cpu()))

def save_acc(descriptor, iou_overlap, results_file, acc):
    with open(f"{MOMENTS_DIR}{args.model_desc}/{results_file}.csv", 'a') as f:
        writer = csv.writer(f)
        writer.writerow([descriptor, round(iou_overlap, 3), round(acc, 3)])

def predict_dataset(data_module, model, trainer, loaders):
    data_module.batch_size = PREDICT_BATCH_SIZE[args.dataset_code]
    cut = data_module.batch_size if args.dev_run else None

    accs = {}

    keys = data_module.keys
    keys.remove('train')

    if args.lit_code == DET_CODE:
         for key in keys:
            preds = trainer.predict(model, loaders[key])
            preds = torch.stack(preds)
            preds = torch.flatten(preds, start_dim=0, end_dim=1)
            acc = model.accuracy(torch.Tensor(preds), torch.Tensor(targets[key]))
            accs[key] = round(acc, 4)
    else:

        if not model.moment:
            model.moment = args.moments
        if model.moment:
            preds, epis, accs = model.predict_custom()

            if args.dataset_code == BMNIST_CODE:
                indices = data_module.test_set.get_indices(BMNIST_BIAS_VARS)
                
                test_targets = model.targets['test']

                print("Maj/min breakdown accuracies:")
                print(model.manual_device)
                for bias_var in BMNIST_BIAS_VARS:

                    maj_min_accs = []

                    for split in ['_maj', '_min']:
                        split_preds = torch.Tensor(preds['test'][indices[bias_var + split]])
                        split_targets = torch.Tensor(test_targets[indices[bias_var + split]]).to(model.manual_device)

                        # print(split_preds.shape)
                        # print(split_targets.shape)
                        split_acc = model.accuracy(
                            split_preds,
                            split_targets
                        )
                        maj_min_accs.append(split_acc)

                    print(f"   {bias_var} maj/min {round(maj_min_accs[0], 4)}/{round(maj_min_accs[1], 4)}")

            if not args.dev_run:
                for key in keys:
                    save_outputs(preds[key], epis[key], extra=f"_{key}")

    if len(accs):
        print("Accuracies: ")
        print("-"*40)
        for key in keys:
            print(f"   {key} set {round(accs[key], 4)}%")


predict_dataset(data_module, model, trainer, loaders)



print("*"*50)
print(f"Finished with model with desc: {model_desc:>20}")
print()