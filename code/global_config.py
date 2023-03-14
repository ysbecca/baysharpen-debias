from local_config import *



DEFAULT_SEED 			= 32



TB_LOGS_PATH 			= ROOT + "tb_logs/"
DATA_DIR 				= ROOT + "data/"

CONFIG_DIR 				= CODE_ROOT + "configs/"
RESULTS_DIR 			= ROOT + "results/"
MOMENTS_DIR 			= ROOT + "moments/"
FIGURES_DIR 			= ROOT + "figures/"
SWEEPS_DIR 				= ROOT + "sweeps/"


#######################################################
#             Datasets
#######################################################

SIMNET_CODE 			= "simnet"
IMAGENET_CODE 			= "imnet"
IMAGENET100_CODE 		= "imnet100"
SYNBOLS_CODE 			= "synbols"
BMNIST_CODE  			= "bmnist"
BAR_CODE 				= "bar"
COCO_PLACES_CODE   		= "cocoplaces"

NUM_CLASSES = {
	IMAGENET100_CODE: 100,
	IMAGENET_CODE: 1000,
	SIMNET_CODE: 1000,
	SYNBOLS_CODE: 2,
	BMNIST_CODE: 10,
	BAR_CODE: 6,
	COCO_PLACES_CODE: 9,
}

# even batch sizes for stacking
PREDICT_BATCH_SIZE = {
	BAR_CODE: 109,
	COCO_PLACES_CODE: 100,
	BMNIST_CODE: 100,
	SYNBOLS_CODE: 100,
}

BMNIST_BIAS_VARS = [
            'digit_color_ix',
            'digit_scale_ix', 
            'digit_position_ix',
            'texture_ix',
            'texture_color_ix',
            'letter_ix',
            'letter_color_ix'
        ]

WEIGHTS_TO_TRACK 		= 250
CAMS_TO_SHOW 			= 12


BMINST_IMAGE_SIZE		= 160
IMAGENET_IMSIZE			= 224
BAR_IMAGE_SIZE  		= IMAGENET_IMSIZE
COCO_PLACES_ORIG_SIZE 	= 178
COCO_PLACES_IMSIZE      = 64
LARGER_COCO_SIZE        = 160

IMAGENET_STD 			= [0.229, 0.224, 0.225]
IMAGENET_MEAN 			= [0.485, 0.456, 0.406]
DEFAULT_THUMBSIZE		= 224
DEFAULT_QUANTILE 		= 0.01
DEFAULT_IOU 			= 0.04
DEFAULT_MASK_THRESHOLD  = 0.5


IMAGENET_TRAINSIZE 		= 1281167
BMNIST_TRAINSIZE   		= 50000

DECAY_MILESTONES = {
	BMNIST_CODE: [50, 70],
	COCO_PLACES_CODE: [100, 120, 140]
}

CLASS_WEIGHTS = {
	BAR_CODE: [0.83, 0.90, 0.92, 0.86, 0.83, 0.90],
	BMNIST_CODE: [1.0]*NUM_CLASSES[BMNIST_CODE],
	COCO_PLACES_CODE: [1.0]*NUM_CLASSES[COCO_PLACES_CODE],
}


BAR_CLASS_NAMES 		= ['climbing', 'diving', 'fishing', 'pole vaulting', 'racing', 'throwing']


# imagenet
RESNET_CKPT_PTH   		= MOMENTS_DIR + "resnet-checkpoint/resnet18-f37072fd.pth"

CKPTS = {
	BAR_CODE: MOMENTS_DIR + "imnet100_resnet.ckpt",
	BMNIST_CODE: MOMENTS_DIR + "bmnist_resnet.ckpt"
}

MTURKS_RESULTS 			= DATA_DIR + "discover_spurious_features.csv"
SIMNET_DIR  			= DATA_DIR + "salient_imagenet_dataset/"
IMAGENET_DIR 			= DATA_DIR + "ILSVRC2012/"
RANKS_FILE 				= DATA_DIR + "ranks.csv"


RES_DIR 				= DATA_DIR + "res/"

RQ_PICKLE 				= "rq.pkl"
TOPK_PICKLE 			= "topk.pkl"

# Biased Synbols datasets
GRAD_BG 				= "GRD"
IMAGE_BG 				= "IM"

SYNBOL_CHARS 			= ["s", "t"]

SYNBOLS_BASE_DIR 		= DATA_DIR + "synbols_data/base/"
BMNIST_BASE_DIR  		= DATA_DIR + "biased_mnist/"
BAR_BASE_DIR  			= DATA_DIR + "BAR/"
COCO_PLACES_BASE_DIR    = DATA_DIR + "cocoplaces/cocoplaces_vf_9_0.8/"


MASK_SIZE		 		= 40
DEFAULT_SPURIOUS_RATIO	= 0.95

SPURIOUS_MODE_REG 		= "regional"
SPURIOUS_MODE_REG_ST 	= "regional-stationary" # stationary block
SPURIOUS_MODE_TEX 		= "textural"
SPURIOUS_MODE_QUAL 		= "quality"
SPURIOUS_MODE_BW	  	= "black_and_white"


SYNBOLS_HALF_SIZE 		= 1000 # samples per class

# loss types
INTERLEAVING_LOSS 		= "interleave"
WEIGHTED_LOSS 			= "weighted"
SHARPEN_LOSS 			= "sharpen"
PCGRAD_LOSS 			= "pcgrad"

SHARPEN_LOSS_SCALAR 	= 5.


#######################################################
#             Architectures
#######################################################


# Lit models
CSGMCMC_CODE 			= "bayesian"
DET_CODE 				= "deterministic"
SHARPEN_CODE 			= "sharpen"


VGG_CODE 				= "vgg"
ALEX_CODE 				= "alex"
RESNET6_CODE            = "resnet6"
RESNET10_CODE           = "resnet10"
RESNET14_CODE           = "resnet14"
RESNET_CODE 			= "resnet"
RESNET50_CODE 			= "resnet50"
RESNET34_CODE           = "resnet34"


# State dict keys for conv layers
ALEX_NET_KEYS 			= [0, 3, 6, 8, 10]


#######################################################
#             Defaults
#######################################################

DEFAULT_MOMENTS 		= 5 * 3 # 5 moments for 3 total cycles

DEFAULT_CKPT_FREQ 		= 100

DEFAULT_MODEL 			= RESNET_CODE

SCALAR_EPI 				= 5.

DEFAULT_OPTIM 			= "sgd"

DEFAULT_MAX_EPOCH 		= 200

DEFAULT_BATCH_SIZE 		= 12 if IS_LOCAL else 64

DEFAULT_FWD_BATCH_SIZE 	= 12 if IS_LOCAL else 128

DEFAULT_NUM_WORKERS 	= 2 if IS_LOCAL else 8

DEFAULT_MOMENTUM 		= 0.9

DEFAULT_LR 				= 0.5

DEFAULT_CYCLE_LENGTH 	= 100

MODELS_PER_CYCLE 		= 5

DEFAULT_CYCLES 			= 4

DEFAULT_RESULT_FILE 	= RESULTS_DIR + "results.txt"

GPU_COUNT 				= 0 if IS_LOCAL else 1



#######################################################
#             Mappings
#######################################################

DATAMODULE_MAP = {
	IMAGENET_CODE: "DataModuleImageNet",
	SYNBOLS_CODE: "DataModuleSpuriousSynbols",
	BMNIST_CODE: "DataModuleBiasedMNIST",
	COCO_PLACES_CODE: "DataModuleCocoPlaces",
	BAR_CODE: "DataModuleBAR",
}