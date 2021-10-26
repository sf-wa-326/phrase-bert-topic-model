"""

Modified from:
https://gist.github.com/SandroLuck/d04ba5c2ef710362f2641047250534b2#file-pytorch-lightning-machine-learning-zero-to-hero-in-75-lines-of-code-py

"""


import os, sys

# allow importing from parent dir
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import torch
import torch.nn.functional as F
from pytorch_lightning import seed_everything, LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping
from torch import nn, optim, rand, sum as tsum, reshape, save
from torch.utils.data import DataLoader, Dataset
import json
from sentence_transformers import SentenceTransformer
from sentence_transformers import models
from utils.spanPooling import spanPooling
import numpy as np
import pickle
from config.model_path import MODEL_PATH
from utils.cls_utils import ParaphraseDataset, ProbingModel, get_data_emb


def main(args):
    device = f'cuda:{args.device_id}'

    model_name_acc_dict = {
        k:0.0 for k in MODEL_PATH.keys()
    }
    for model_name, model_path in MODEL_PATH.items():
        if 'ppdb' in args.task: 
            # for ppdb dataset, we split the constructed the dataset and store the train / val / test splits in cache
            data_fname = os.path.join( args.data_dir, 'examples.json' )
            phrase1_tensor, phrase2_tensor, label_tensor = get_data_emb(args.full_run_mode, data_fname, model_path, device)
            phrase1_tensor.to(device)
            phrase2_tensor.to(device)
            label_tensor.to(device)
            split1 = math.ceil( phrase1_tensor.size()[0] * 0.7 )
            split2 = math.ceil( phrase1_tensor.size()[0] * 0.85 )

            train_dataset = ParaphraseDataset( phrase1_tensor[:split1, :], 
                                            phrase2_tensor[:split1, :], 
                                            label_tensor[:split1]  )
            valid_dataset = ParaphraseDataset( phrase1_tensor[split1:split2, :], 
                                            phrase2_tensor[split1:split2, :], 
                                            label_tensor[split1:split2]  )
            test_dataset = ParaphraseDataset(  phrase1_tensor[split2:, :], 
                                            phrase2_tensor[split2:, :], 
                                            label_tensor[split2:])
        elif 'paws' in args.task: 
            # for paws dataset, we use the train / val / test split defined by the authors
            data_fname = os.path.join( args.data_dir, 'train_examples.json' )
            phrase1_tensor, phrase2_tensor, label_tensor = get_data_emb(args.full_run_mode, data_fname, model_path, device)
            phrase1_tensor.to(device)
            phrase2_tensor.to(device)
            label_tensor.to(device)
            train_dataset = ParaphraseDataset( phrase1_tensor, phrase2_tensor, label_tensor )

            data_fname = os.path.join( args.data_dir, 'dev_examples.json' )
            phrase1_tensor, phrase2_tensor, label_tensor = get_data_emb(args.full_run_mode, data_fname, model_path, device)
            phrase1_tensor.to(device)
            phrase2_tensor.to(device)
            label_tensor.to(device)
            valid_dataset = ParaphraseDataset( phrase1_tensor, phrase2_tensor, label_tensor )

            data_fname = os.path.join( args.data_dir, 'test_examples.json' )
            phrase1_tensor, phrase2_tensor, label_tensor = get_data_emb(args.full_run_mode, data_fname, model_path, device)
            phrase1_tensor.to(device)
            phrase2_tensor.to(device)
            label_tensor.to(device)
            test_dataset = ParaphraseDataset( phrase1_tensor, phrase2_tensor, label_tensor )
        else:
            print('Not a valid task')

        early_stop_callback = EarlyStopping(monitor='epoch_val_accuracy', min_delta=0.00, patience=5, verbose=True, mode='max')
        model = ProbingModel( input_dim=phrase1_tensor.shape[1] * 2, 
                        train_dataset=train_dataset, 
                        valid_dataset=valid_dataset,
                        test_dataset=test_dataset ).to(device)
        trainer = Trainer(max_epochs=100, min_epochs=3, auto_lr_find=False, auto_scale_batch_size=False,
                        progress_bar_refresh_rate=10, callbacks=[early_stop_callback], gpus=[args.device_id])
        # trainer.tune(model)
        trainer.fit(model)
        result = trainer.test(test_dataloaders=model.test_dataloader())
        print(f'\n finished {model_name}\n')
        output_fname = os.path.join( args.result_dir, f'{args.task}_{model_name}.json' )

        with open(output_fname, 'w') as f:
            json.dump( result, f, indent=4)
        
        model_name_acc_dict[model_name] = result[0]['epoch_test_accuracy']
        v = model_name_acc_dict[model_name]
        print(result)
        print()

    # print(str( model_name_acc_dict))
    for k,v in model_name_acc_dict.items():
        print(f' model: {k}, testing accuracy: {v:.4f} ')

    print('Done with main')

if __name__ == '__main__':
    seed_everything(42)
    import argparse
    import math 
    parser = argparse.ArgumentParser()
    parser.add_argument("--full_run_mode", action='store_true')
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--task", type=str, default='paws_short', 
                    choices=['ppdb_exact', 'ppdb', 'paws_short'])
    parser.add_argument("--data_dir", type=str, default='')
    parser.add_argument("--result_dir", type=str, default='')

    args = parser.parse_args()
    main(args)

