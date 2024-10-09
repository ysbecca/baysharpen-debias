# baysharpen-debias

Code relevant for work present in thesis "Visual bias mitigation driven by Bayesian uncertainties".

## Setup

This code assumes additional directories in default locations:
- ```tb_logs/```: Tensorboard log directory
- ```data/```: datasets

Set the path directories in ```local_config.py```.

All default configurations, paths, and parameters can be changed in ```global_config.py```.

## Datasets

Links to download datasets:

+ [Biased MNIST](https://github.com/erobic/occam-nets-v1)
+ [COCO-on-Places](https://github.com/Faruk-Ahmed/predictive_group_invariance)
+ [Biased Action Recognition](https://github.com/alinlab/BAR)

## Training

The repository includes checkpoints for baseline Bayesian ResNet18 models trained on the three datasets. Run scripts ```sharpen-bmnist.sh```, ```sharpen-coco.sh```, and ```sharpen-bar.sh``` to obtain sharpening results reported in the paper.

Alternately, to modify parameters, you can run the ```train.py``` Python script directly on the command line:

```
usage: train.py [-h] [--task_id TASK_ID] [--config_name CONFIG_NAME]
                [--dataset_code DATASET_CODE] [--lit_code LIT_CODE]
                [--model_code MODEL_CODE] [--model_desc MODEL_DESC]
                [--ckpt CKPT] [--optim_code OPTIM_CODE]
                [--max_epochs MAX_EPOCHS] [--batch_size BATCH_SIZE]
                [--momentum MOMENTUM] [--lr LR] [--result_file RESULT_FILE]
                [--cycle_length CYCLE_LENGTH] [--num_gpus NUM_GPUS]
                [--dev_run DEV_RUN] [--use_vw_flag_coco USE_VW_FLAG_COCO]
                [--bmnist_size BMNIST_SIZE] [--small_imagenet SMALL_IMAGENET]
                [--pretrained PRETRAINED] [--keep_fc KEEP_FC] [--alpha ALPHA]
                [--epiwt EPIWT] [--kappa KAPPA] [--moments MOMENTS]
                [--temperature TEMPERATURE] [--cycles CYCLES]
                [--models_per_cycle MODELS_PER_CYCLE] [--bg_mode BG_MODE]
                [--spurious_mode SPURIOUS_MODE]
                [--spurious_ratio SPURIOUS_RATIO] [--sf_size SF_SIZE]
                [--loss_type LOSS_TYPE] [--loss_scalar LOSS_SCALAR] 
                [--moment_desc MOMENT_DESC]
```

All configs YAML files are under ```configs/``` and any hyperparameters not specified there assume default values specified in ```global_config.py``` or provided on command line as arg variables.

## Checkpoints

Checkpoints for the Bayesian posterior estimates for Bayesian baseline models on the three datasets are included in ```moments/```. These include all $\theta_m$ in $\Theta$ and predictions on all test subsets.

Checkpoints for final sharpened posterior models are also included.

## Evaluation

To evaluate and generate predictions for the sharpened BayResNet+Sharpen, run:

```python train.py --config_name eval --task_id DATASET_TASK_ID```

where ```DATASET_TASK_ID``` is 0 (Biased MNIST), 1 (COCO-on-Places), or 2 (BAR).

## Environment

See ```requirements.txt```.

## Citation

To be included after review period.

## Acknowledgements

We would like to acknowledge the respective authors for their publicly available code for [cSG-MCMC](https://github.com/ruqizhang/csgmcmc), [network dissection](https://github.com/davidbau/dissect), [PCGrad](https://github.com/WeiChengTseng/Pytorch-PCGrad), [OccamNets](https://github.com/erobic/occam-nets-v1) whose code has been partially used and modified for this work.
