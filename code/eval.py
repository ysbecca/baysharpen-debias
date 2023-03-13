import sys
import os
import numpy as np
import math
import pandas as pd
import pytorch_lightning as pl

sys.path.append(os.path.abspath('..'))

import argparse
import torch
import random

from dataset_models.simnet import *
from dataset_models.simnet_dm import *

from litmodels.litsgmcmc import *
from utils import *
from global_config import *



sns.set_theme()


# Set up training arguments and parse
parser = argparse.ArgumentParser(description='training network')

parser.add_argument(
	'--model_desc', type=str, default="test",
	help='model description; if not defined, set as test')
parser.add_argument(
	'--dev_run', type=bool, default=False,
	help='runs a dev testing run on limited num of batches')

args = parser.parse_args()


print("=======================================")
print("		 EVALUATION PARAMS			 ")
print("=======================================")
for arg in vars(args):
	print(F"{arg:>20} {getattr(args, arg)}")



def get_wordnet_id_from_srank(n, df, targets):
	return df[df['spurious'] == n]['wordnet_id']

def acc_for_class_id(c):
	""" Compute the mean accuracy for a given class """
	# indices of elements for class instances
	targets_c = [i for i, t in enumerate(targets) if t == c]
	# compute accuracy
	if len(targets_c):
		acc = (np.array(preds)[targets_c] == np.array([c]*len(targets_c))).sum()
		acc = round(acc / len(targets_c) * 100., 3)
	else:
		acc = 0

	return acc


def stats_for_s_rank(s, epis, preds, df, targets):
	""" Computes mean and std deviation for all classes of a given S-rank """

	wordnet_ids_s = get_wordnet_id_from_srank(s, df, targets)

	accs = []
	epi_list = []
	for index in list(wordnet_ids_s.index):
		accs.append(acc_for_class_id(index))
		epi_list.append(epis[index][preds[index]]) # epistemic uncertainty of true class
	if len(wordnet_ids_s) == 0:
		mean = 0
		epi_mean = 0
	else:
		mean = round(sum(accs) / len(wordnet_ids_s), 3)
		epi_mean = round(sum(epi_list) / len(wordnet_ids_s), 3)
	return mean, np.std(np.array(accs)), epi_mean, np.std(np.array(epi_list))



#######################################################
#			 Dataset and prediction prep
#######################################################

# load dataset
simnet = SalientImageNetComplete(transform=eval_transform)
simnet.setup()

cutoff = DEFAULT_BATCH_SIZE if args.dev_run else len(simnet)

targets = np.array(simnet.class_ids)[:cutoff]

# load preds and uncertainties
char = "_dev" if args.dev_run else ""

# load model preds and uncertainties
preds = np.load(f"{MOMENTS_DIR}{args.model_desc}/preds{char}.npy")
epis = np.load(f"{MOMENTS_DIR}{args.model_desc}/epis{char}.npy")

print("preds.shape", preds.shape)
print("epis.shape ", epis.shape)

# load the csv for MTurk results
df = pd.read_csv(RANKS_FILE)
print(df.head())


#######################################################
#			 Compute statistics
#######################################################

stats = []
for i in range(6):
	stats.append(stats_for_s_rank(i, epis, preds, df, targets))


#######################################################
#			 Plot the rank stats and save
#######################################################

plot_stats(stats, args.model_desc, save=(not args.dev_run))












