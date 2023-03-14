import torch
import numpy as np


from global_config import *



def save_outputs(preds, epis, extra=""):
    """ Saves predictions and epistemic uncertainites in numpy arrays. """

    np.save(f"{MOMENTS_DIR}{model_desc}/preds{char}{extra}.npy", np.argmax(preds.detach().cpu(), axis=1))
    np.save(f"{MOMENTS_DIR}{model_desc}/epis{char}{extra}.npy", np.array(epis.detach().cpu()))



def predict_dataset(data_module, model, trainer, loaders, args):
    """ Generates predictions and computes accuracy on all test sets for
        a given data module.
    """

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
