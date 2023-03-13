# Posterior Estimate Sharpening

Code for paper "Implicit Visual Bias Mitigation by Posterior Estimate Sharpening of a Bayesian Neural Network"

## Setup

This code assumes additional directories in default locations:
- ```tb_logs/```: Tensorboard log directory
- ```data/```: datasets

All default configurations, paths, and parameters can be changed in ```global_config.py```.

## Datasets

Links to download datasets are below:

+ [Biased MNIST](https://github.com/erobic/occam-nets-v1)
+ [COCO-on-Places](https://github.com/Faruk-Ahmed/predictive_group_invariance)
+ [Biased Action Recognition](https://github.com/alinlab/BAR)

## Training

Run scripts ```sharpen-bmnist.sh```, ```sharpen-coco.sh```, and ```sharpen-bar.sh``` to obtain sharpening results reported in the paper.

Alterately, to change parameters, you can run the ```train.py``` Python script directly on the command line:

```python train.py --config_name baybar --task_id 10 --model_desc bar-sharpen-test```

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

We would like to acknowledge the respective authors for their publicly available code for [cSG-MCMC](https://github.com/ruqizhang/csgmcmc), [network dissection](https://github.com/davidbau/dissect), and [OccamNets](https://github.com/erobic/occam-nets-v1) whose code has been partially used and modified for this work.
