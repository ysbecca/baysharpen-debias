"""


Compute and save UNIT uncertainties for each unit in a given Bayesian model.


"""


"""

Dissect the network as per Bau PNAS 2020 using SalientImageNet for concepts and my Bayesian network.


"""

import yaml
import sys
import os
import numpy as np
import math
import pickle
from os.path import exists
import argparse

sys.path.append(os.path.abspath('..'))

import torch
import torchvision.models as models

from utilmodels.tally import *
from utilmodels.visualizers import *
from nnmodels.nethook import *
from nnmodels.resnet import *
from nnmodels.alexnet import *

from utilmodels import pbar, renormalize, pidfile
from utilmodels import upsample, tally, imgviz, imgsave, bargraph


from dataset_models.simnet_dm import *
from dataset_models.synbols_dm import *

from global_config import *


# set up training arguments and parse
parser = argparse.ArgumentParser(description='analyse unit uncertainty of Bayesian network')


parser.add_argument(
    '--model_desc', type=str, default="synbolsG-reg",
    help='model description; if not defined, set as test')
parser.add_argument(
    '--model_code', type=str, default=ALEX_CODE,
    help='model architecture')
parser.add_argument(
    '--dev_run', type=bool, default=False,
    help='runs a dev testing quick run')
parser.add_argument(
    '--moments', type=int, default=12,
    help='moments')

args = parser.parse_args()



print("=======================================")
print("     UNCERTAINTY ANALYSIS PARAMS       ")
print("=======================================")
for arg in vars(args):
    print(F"{arg:>20} {getattr(args, arg)}")


# load the saved model; go through the state dict? AlexNet manually first to understand 
device = torch.device('cpu')
moments = 2 if args.dev_run else args.moments

states = []
for m in range(moments):
	
	m_path = f"{MOMENTS_DIR}{args.model_desc}/moment_{m}.pt"
	state_dict = torch.load(m_path, map_location=device)

	states.append(state_dict)


assert args.model_code == ALEX_CODE

""" 
Classification part of network
"""

fc_uncs = dict()

for key in [1, 4, 6]:
	weights_key = f"classifier.{key}.weight"
	bias_key = f"classifier.{key}.bias"

	stacked_w = torch.stack([s[weights_key] for s in states])
	stacked_b = torch.stack([s[bias_key] for s in states])

	fc_uncs[weights_key] = stacked_w
	fc_uncs[bias_key] = stacked_b

pickle.dump(fc_uncs, open(f"{MOMENTS_DIR}{args.model_desc}/fc_uncs.pkl", "wb"))


"""
Feature section of network

"""

cumulative_unit_unc = dict()
unit_unc = dict()

# for each unit in each layer
for key in ALEX_NET_KEYS:
	weights_key = f"features.{key}.weight"
	bias_key = f"features.{key}.bias"

	# print(f"stacked {weights_key}")
	stacked_w = torch.stack([s[weights_key] for s in states])
	stacked_b = torch.stack([s[bias_key] for s in states])

	# compute the unit variance (unit uncertainty)
	# [num_kernels, out_channels, h, w]
	unit_unc[weights_key] 		     = torch.var(stacked_w, dim=0)
	# [bias_dim]
	unit_unc[bias_key] 		 	     = torch.var(stacked_b)

	# cumulative kernel uncertainty
	# [num_kernels]
	cumulative_unit_unc[weights_key] = torch.var(stacked_w, dim=(0,2,3,4))


# save.
uncertainties = {
	"cumulative_unit_unc": cumulative_unit_unc,
	"unit_unc": unit_unc,
}

pickle.dump(uncertainties, open(f"{MOMENTS_DIR}{args.model_desc}/unit_uncs.pkl", "wb"))







