import torch
import torch.nn.functional as F
from pytorch_lightning import seed_everything, LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping
from torch import nn, optim, rand, sum as tsum, reshape, save
from torch.utils.data import DataLoader, Dataset
import json
from sentence_transformers import SentenceTransformer
from sentence_transformers import models
import numpy as np
from utils.utils import load_model
from utils.glove_utils import get_phrase_emb, get_word_emb
from utils.spanPooling import spanPooling


def get_data_emb(full_run_mode, data_fname, model_path, device):
    if 'glove' in model_path:
        return get_glove_data_emb(full_run_mode, data_fname, model_path, device)


    with open(data_fname, 'r') as f:
        data_list = json.load(f)
    
    phrase1_list = [ item[0] for item in data_list ] 
    phrase2_list = [ item[1] for item in data_list ]
    label = [ item[2] for item in data_list]

    if not full_run_mode:
        subset_size = 50
        phrase1_list = phrase1_list[:subset_size]
        phrase2_list = phrase2_list[:subset_size]
        label = label[:subset_size]
    
    model = load_model(model_path, device)
    model.to(device)
    print(device)
    emb_batch_size = 8

    phrase1_emb_tensor_list = encode_in_batch(model, emb_batch_size, phrase1_list)
    phrase2_emb_tensor_list = encode_in_batch(model, emb_batch_size, phrase2_list)

    label_list = [ 1 if e == 'pos' else 0 for e in label ]

    import random
    random.seed(42)
    combined = list( zip(phrase1_emb_tensor_list, phrase2_emb_tensor_list, label_list))
    random.shuffle( combined )
    phrase1_emb_tensor_list_shuffled, phrase2_emb_tensor_list_shuffled, label_list_shuffled = zip(*combined)
    label_tensor = torch.FloatTensor( label_list_shuffled)
    
    return torch.stack(phrase1_emb_tensor_list_shuffled), torch.stack(phrase2_emb_tensor_list_shuffled), label_tensor



def get_glove_data_emb(full_run_mode, data_fname, model_path, device):
    with open(data_fname, 'r') as f:
        data_list = json.load(f)
    
    phrase1_list = [ item[0] for item in data_list ] 
    phrase2_list = [ item[1] for item in data_list ]
    label = [ item[2] for item in data_list]

    if not full_run_mode:
        subset_size = 50
        phrase1_list = phrase1_list[:subset_size]
        phrase2_list = phrase2_list[:subset_size]
        label = label[:subset_size]
    
    word2coef_dict, average_emb = load_model(model_path, device)

    emb_list = []
    for term in phrase1_list:
        emb = get_phrase_emb(word2coef_dict, term, average_emb)
        emb_list.append(emb)
    phrase1_emb_tensor_list = torch.tensor(emb_list, dtype=torch.float32)
    emb_list = []
    for term in phrase1_list:
        emb = get_phrase_emb(word2coef_dict, term, average_emb)
        emb_list.append(emb)
    phrase2_emb_tensor_list = torch.tensor(emb_list, dtype=torch.float32)
    label_list = [ 1 if e == 'pos' else 0 for e in label ]

    import random
    random.seed(42)
    combined = list( zip(phrase1_emb_tensor_list, phrase2_emb_tensor_list, label_list))
    random.shuffle( combined )
    phrase1_emb_tensor_list_shuffled, phrase2_emb_tensor_list_shuffled, label_list_shuffled = zip(*combined)
    label_tensor = torch.FloatTensor( label_list_shuffled)

    
    return torch.stack(phrase1_emb_tensor_list_shuffled), torch.stack(phrase2_emb_tensor_list_shuffled), label_tensor

def encode_in_batch(model, batch_size, text_list):
    all_emb_tensor_list = []
    for i in range( 0, len(text_list), batch_size ):
        batch_text_list = text_list[i:i+batch_size]
        batch_emb_list = model.encode( batch_text_list, batch_size=batch_size, convert_to_tensor=True, show_progress_bar=False)
        all_emb_tensor_list.extend( batch_emb_list )
    return all_emb_tensor_list


class ParaphraseDataset(Dataset):
    def __init__(self, phrase1_tensor, phrase2_tensor, label_tensor ):
        self.concat_input = torch.cat( (phrase1_tensor, phrase2_tensor), 1 )
        self.label = label_tensor

    def __getitem__(self, index):
        return (self.concat_input[index], self.label[index])

    def __len__(self):
        return self.concat_input.size()[0]


class ProbingModel(LightningModule):
    def __init__(self, input_dim=1536, train_dataset=None, valid_dataset=None, test_dataset=None):
        super(ProbingModel, self).__init__()
        # Network layers
        self.input_dim = input_dim
        self.linear = nn.Linear(self.input_dim, 256)
        self.linear2 = nn.Linear(256, 1)
        self.output = nn.Sigmoid()

        # Hyper-parameters, that we will auto-tune using lightning!
        self.lr = 0.0001
        self.batch_size = 200

        # datasets
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset

    def forward(self, x):
        x1 = self.linear(x)
        x1a = F.relu(x1)
        x2 = self.linear2(x1a)
        output = self.output(x2)
        return reshape(output, (-1,))

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

    def train_dataloader(self):
        loader = DataLoader( self.train_dataset, batch_size=self.batch_size, shuffle=True)
        return loader
    
    def val_dataloader(self):
        loader = DataLoader( self.valid_dataset, batch_size=self.batch_size, shuffle=False)
        return loader

    def test_dataloader(self):
        loader = DataLoader( self.test_dataset, batch_size=self.batch_size, shuffle=False)
        return loader


    def compute_accuracy(self, y_hat, y):
        with torch.no_grad():
            y_pred = ( y_hat >= 0.5 )
            y_pred_f = y_pred.float()
            num_correct = tsum( y_pred_f == y )
            denom = float(y.size()[0])
            accuracy = torch.div( num_correct, denom)
        return accuracy

    def training_step(self, batch, batch_nb):
        mode = 'train'
        x, y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy(y_hat, y)
        accuracy = self.compute_accuracy(y_hat, y)
        self.log(f'{mode}_loss', loss, on_epoch=True, on_step=True)
        self.log(f'{mode}_accuracy', accuracy, on_epoch=True, on_step=True)
        return {f'loss': loss, f'{mode}_accuracy':accuracy, 'log': {f'{mode}_loss': loss}}


    def training_epoch_end(self, outputs):
        mode = 'train'
        loss_mean = sum([o[f'loss'] for o in outputs]) / len(outputs)
        accuracy_mean = sum([o[f'{mode}_accuracy'] for o in outputs]) / len(outputs)
        self.log(f'epoch_{mode}_loss', loss_mean, on_epoch=True, on_step=False)
        print(f'\nThe end of epoch {mode} loss is {loss_mean.item():.4f}')
        self.log(f'epoch_{mode}_accuracy', accuracy_mean, on_epoch=True, on_step=False)
        print(f'\nThe end of epoch {mode} accuracy is {accuracy_mean.item():.4f}')


    def validation_step(self, batch, batch_nb):
        mode = 'val'
        x, y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy(y_hat, y)
        accuracy = self.compute_accuracy(y_hat, y)
        self.log(f'{mode}_loss', loss, on_epoch=True, on_step=True)
        self.log(f'{mode}_accuracy', accuracy, on_epoch=True, on_step=True)
        return {f'{mode}_loss': loss, f'{mode}_accuracy':accuracy, 'log': {f'{mode}_loss': loss}}


    def validation_epoch_end(self, outputs):
        mode = 'val'
        loss_mean = sum([o[f'{mode}_loss'] for o in outputs]) / len(outputs)
        accuracy_mean = sum([o[f'{mode}_accuracy'] for o in outputs]) / len(outputs)
        self.log(f'epoch_{mode}_loss', loss_mean, on_epoch=True, on_step=False)
        print(f'\nThe end of epoch {mode} loss is {loss_mean.item():.4f}')
        self.log(f'epoch_{mode}_accuracy', accuracy_mean, on_epoch=True, on_step=False)
        print(f'\nThe end of epoch {mode} accuracy is {accuracy_mean.item():.4f}')

    def test_step(self, batch, batch_nb):
        mode = 'test'
        x, y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy(y_hat, y)
        accuracy = self.compute_accuracy(y_hat, y)
        self.log(f'{mode}_loss', loss, on_epoch=True, on_step=True)
        self.log(f'{mode}_accuracy', accuracy, on_epoch=True, on_step=True)
        return {f'{mode}_loss': loss, f'{mode}_accuracy':accuracy, 'log': {f'{mode}_loss': loss}}

    def test_epoch_end(self, outputs):
        mode = 'test'
        loss_mean = sum([o[f'{mode}_loss'] for o in outputs]) / len(outputs)
        accuracy_mean = sum([o[f'{mode}_accuracy'] for o in outputs]) / len(outputs)
        self.log(f'epoch_{mode}_loss', loss_mean, on_epoch=True, on_step=False)
        print(f'\nThe end of epoch {mode} loss is {loss_mean.item():.4f}')
        self.log(f'epoch_{mode}_accuracy', accuracy_mean, on_epoch=True, on_step=False)
        print(f'\nThe end of epoch {mode} accuracy is {accuracy_mean.item():.4f}')
