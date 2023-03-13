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
parser = argparse.ArgumentParser(description='dissect network')

parser.add_argument(
    '--dataset_code', type=str, default=IMAGENET_CODE,
    help='the dataset string code')
parser.add_argument(
    '--model_code', type=str, default=None,
    help='model code for model architecture to use')
parser.add_argument(
    '--model_desc', type=str, default="test",
    help='model description; if not defined, set as test')
parser.add_argument(
    '--batch_size', type=int, default=DEFAULT_FWD_BATCH_SIZE,
    help='training batch size')

parser.add_argument(
    '--num_gpus', type=int, default=1,
    help='number of gpus')
parser.add_argument(
    '--dev_run', type=bool, default=False,
    help='runs a dev testing run on limited num of batches')
parser.add_argument(
    '--save_unit_images', type=bool, default=False,
    help='save unit images')
parser.add_argument(
    '--compute_ious', type=bool, default=False,
    help='use segmenter and compute ious at percentile')


parser.add_argument(
    '--moments', type=int, default=DEFAULT_MOMENTS,
    help='for testing only on pre-trained model; set manually')

parser.add_argument(
    '--image_size', type=int, default=IMAGENET_IMSIZE,
    help='size of dataset images')
parser.add_argument(
    '--thumbsize', type=int, default=DEFAULT_THUMBSIZE,
    help='thumbnail size for unit activation maps')
parser.add_argument(
    '--quantile', type=float, default=DEFAULT_QUANTILE,
    help='quantile')
parser.add_argument(
    '--iou_threshold', type=float, default=DEFAULT_IOU,
    help='min iou threshold')
parser.add_argument(
    '--mask_threshold', type=float, default=DEFAULT_MASK_THRESHOLD,
    help='soft mask threshold')
parser.add_argument(
    '--num_workers', type=int, default=DEFAULT_NUM_WORKERS,
    help='number of workers')


parser.add_argument(
    '--bg_mode', type=str, default=GRAD_BG,
    help='for spurious synbols: background mode')
parser.add_argument(
    '--spurious_mode', type=str, default=None,
    help='for spurious synbols: type of spurious bias')
parser.add_argument(
    '--spurious_ratio', type=float, default=DEFAULT_SPURIOUS_RATIO,
    help='for spurious synbols: ratio for class samples w spurious feature')
parser.add_argument(
    '--layer', type=str, default=None,
    help='layer to dissect')



args 			= parser.parse_args()
percent_level 	= 1.0 - args.quantile

print("=======================================")
print("         DISSECT PARAMS                ")
print("=======================================")
for arg in vars(args):
    print(F"{arg:>20} {getattr(args, arg)}")


default_layer = None
# for testing, model_code = None
if args.model_code == None:
	default_layer = "conv1"
	model = models.resnet152(pretrained=True)
elif args.model_code == RESNET_CODE:
	model = ResNet18(num_classes=NUM_CLASSES[args.dataset_code])
elif args.model_code == RESNET50_CODE:
	model = ResNet50(num_classes=NUM_CLASSES[args.dataset_code])
elif args.model_code == ALEX_CODE:
	default_layer = "features.10"
	model = alexnet(num_classes=NUM_CLASSES[args.dataset_code])


the_layer = args.layer if args.layer else default_layer
print(f"Dissecting all units from layer {the_layer} -----------")



model_dir = f"{MOMENTS_DIR}{args.model_desc}/"
respath = pidfile.exclusive_dirfn(f"{MOMENTS_DIR}{args.model_desc}/{the_layer}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




model = InstrumentedModel(model).to(device).eval()


# add hook to the layer
model.retain_layer(the_layer)


def make_upfn(args, dataset, model, layername):
    '''Creates an upsampling function.'''
    convs, data_shape = None, None

    _ = model(dataset[0][0].unsqueeze(dim=0).to(device))
    data_shape = model.retained_layer(layername).shape[2:]
    pbar.print('upsampling from data_shape', tuple(data_shape))
    upfn = upsample.upsampler(
            (args.thumbsize, args.thumbsize),
            data_shape=data_shape,
            source=dataset,
            convolutions=convs)
    return upfn


if args.dataset_code == SIMNET_CODE:
    data_module = DataModuleSalientImageNet(
        batch_size=args.batch_size,
        dev_run=args.dev_run,
        mask_threshold=args.mask_threshold,
    )
    data_module.setup()
    dataset = data_module.simnet


elif args.dataset_code == SYNBOLS_CODE:
    data_module = DataModuleSpuriousSynbols(
        batch_size=args.batch_size,
        seg_mode=False,
        args=args,
    )

    data_module.setup()
    dataset = data_module.valid_set
else:
    print("[ERR]: no recognised data code")
    pass


torch.set_grad_enabled(False)


upfn = make_upfn(args, dataset, model, the_layer)


def fetch_bay_acts(x, model, layer):
	""" fetches the sampled activations from retained layer """

	m_acts = []
	for m in range(args.moments):
		moment_path = f"{MOMENTS_DIR}{args.model_desc}/moment_{m}.pt"
		model.model.load_state_dict(
			torch.load(moment_path, map_location=device)
		)
		model.eval()
		# print(f"[INFO] Loaded moment {m}")

		_ = model(x)

		# torch.Size([batch_size, 64, 112, 112])
		acts = model.retained_layer(layer)
		m_acts.append(acts)

	# torch.Size([moments, batch_size, 64, 112, 112])
	m_acts = torch.stack(m_acts, dim=0)
	# torch.Size([batch_size, 64, 112, 112])
	m_acts = m_acts.mean(dim=0)
	return m_acts


def compute_samples(batch, *args):
	x = batch.to(device)

	acts = fetch_bay_acts(x, model, the_layer)
	hacts = upfn(acts)
	return hacts.permute(0, 2, 3, 1).contiguous().view(-1, acts.shape[1])


# try to read cached version
print("tally quantiles")
rq = tally.tally_quantile(
	compute_samples,
	dataset,
	sample_size=len(dataset),
	num_workers=args.num_workers,
	pin_memory=True,
	batch_size=args.batch_size,
	cachefile=respath('rq.npz')
)


def compute_image_max(batch, *args):
	x = batch.to(device)
	# _ = model(x)
	# acts = model.retained_layer(the_layer)
	acts = fetch_bay_acts(x, model, the_layer)

	acts = acts.view(acts.shape[0], acts.shape[1], -1)
	acts = acts.max(2)[0]
	return acts


print("tally top k")

topk = tally.tally_topk(
	compute_image_max,
	dataset,
	sample_size=len(dataset),
	batch_size=args.batch_size,
	num_workers=args.num_workers,
	pin_memory=True,
	cachefile=respath('topk.npz')
)


# visualize top-activating patches of top-activating images.
print("visualize top-activating patches of top-activating images")
image_size, image_source = None, None
image_source = dataset


iv = imgviz.ImageVisualizer((args.thumbsize, args.thumbsize),
    image_size=image_size,
    source=dataset,
    quantiles=rq,
    level=rq.quantiles(percent_level))


def compute_acts(data_batch, *ignored_class):
	x = data_batch.to(device)
	out_batch = model(x)
	# acts_batch = model.retained_layer(the_layer)
	acts_batch = fetch_bay_acts(x, model, the_layer)

	return (acts_batch, x)



if args.save_unit_images:
	image_row_width = 6

	unit_images = iv.masked_images_for_topk(
	        compute_acts, dataset, topk,
	        k=image_row_width, num_workers=args.num_workers, pin_memory=True,
	        cachefile=respath('top%dimages.npz' % image_row_width))

	print("saving images")
	imgsave.save_image_set(unit_images, respath(f"images/unit%d.jpg"),
	        sourcefile=respath('top%dimages.npz' % image_row_width))


if args.compute_ious:

	if args.dataset_code == SIMNET_CODE:
		# return seg masks with fids only - reload data in seg mode
		mask_simnet_dm = DataModuleSalientImageNet(
			batch_size=args.batch_size,
			seg_mode=True,
			dev_run=args.dev_run,
			mask_threshold=args.mask_threshold,
		)
		mask_simnet_dm.setup()
		mask_dataset = mask_simnet_dm.simnet
	elif args.dataset_code == SYNBOLS_CODE:

	    data_module = DataModuleSpuriousSynbols(
	        batch_size=args.batch_size,
	        seg_mode=True,
	        args=args,
	    )

	    data_module.setup()
	    mask_dataset = data_module.valid_set

	# Compute IoU agreement between segmentation labels and every unit
	# Grab the 99th percentile, and tally conditional means at that level.
	level_at_99 = rq.quantiles(percent_level).cuda()[None,:,None,None]

	# segmodel, seglabels, segcatlabels = setting.load_segmenter(args.seg)
	# renorm = renormalize.renormalizer(dataset, target='zc')
	def compute_conditional_indicator(batch, *args):
	   
	    # load batch of images and their salient masks
	    # seg is a tensor assigning semantic segmentation labels to every
	    # pixel of an image seg[i, 0] is 0th label for each pixel of ith image
	    # (N, fid, y, x)
	    seg = batch.to(device)
	    batch = args[0].to(device)
	    _ = model(batch)
	    # acts_batch = model.retained_layer(the_layer)

	    acts = fetch_bay_acts(batch, model, the_layer)
	    
	    # image_batch = out_batch if is_generator else renorm(data_batch)
	    # seg = segmodel.segment_batch(image_batch, downsample=4)
	    # acts = model.retained_layer(the_layer)

	    hacts = upfn(acts)
	    iacts = (hacts > level_at_99).float() # indicator
	    return tally.conditional_samples(iacts, seg)


	pbar.descnext('condi99')
	condi99 = tally.tally_conditional_mean(
	    	compute_conditional_indicator,
	        mask_dataset,
	        sample_size=len(dataset),
	        num_workers=args.num_workers,
	        pin_memory=True,
	        batch_size=args.batch_size,
	        cachefile=respath('condi99.npz')
	)

	# Now summarize the iou stats and graph the units
	iou_99 = tally.iou_from_conditional_indicator_mean(condi99)

	pickle.dump(iou_99, open(f"{MOMENTS_DIR}{args.model_desc}/{the_layer}/iou_99.pkl", "wb"))

# unit_label_99 = [
#         (concept.item(), seglabels[concept],
#             segcatlabels[concept], bestiou.item())
#         for (bestiou, concept) in zip(*iou_99.max(0))]
# labelcat_list = [labelcat
#         for concept, label, labelcat, iou in unit_label_99
#         if iou > args.iou_threshold]


print('done')






