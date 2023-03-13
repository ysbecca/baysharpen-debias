import yaml

import matplotlib.pyplot as plt
import seaborn as sns

from global_config import *



def override_from_config(args):

	if args.config_name:
		path = CONFIG_DIR + args.config_name + ".yml"
		args_copy = args

		stream = open(path, 'r')
		cfg_dict = yaml.safe_load(stream)[args.task_id]
		args_dict = vars(args_copy)

		for key, value in cfg_dict.items():
			if key in args_dict:
				# set any which have been found in the yml
				args_dict[key] = value

		return args_copy
	else:
		print("[WARN]: no config file set, no overwriting")

		return args


def plot_stats(stats, model_desc, save=False):
	""" Plots the mean accuracy, uncertainty, and std dev by S-rank """

	fig, ax1 = plt.subplots()
	ax2 = ax1.twinx()

	ax1.errorbar(
	    range(6),
	    [x[0] for x in stats],
	    yerr=[x[1] for x in stats],
	    marker='o',
	    color='darkgray',
	    label="mean accuracy"
	)

	ax2.errorbar(
	    range(6),
	    [x[2] for x in stats],
	    yerr=[x[3] for x in stats],
	    marker='o',
	    color='mediumvioletred',
	    label="mean epistemic unc"
	)

	plt.title("Mean S-rank accuracy")
	ax1.set_xlabel('S-rank')


	ax1.set_ylabel('Mean accuracy')
	ax2.set_ylabel('Mean uncertainty')

	if not save:
		plt.show()
	else:
		plt.savefig(f"{FIGURES_DIR}{model_desc}_stats.pdf")
