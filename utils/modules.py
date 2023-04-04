#!/usr/bin/env python
# coding: utf-8


#Store the PyTorch Lightning Data Modules and Model Classes 

import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule ,Callback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import wandb
from torch.utils.data import Dataset, DataLoader
from torchvision import  utils
import torchvision.transforms as T
import torchvision
import torchvision.models as models
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from typing import Optional
from torchvision import transforms

from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import matplotlib.pyplot as plt

torch.manual_seed(17)

from torchmetrics.classification import Accuracy, Recall
from torchmetrics import Precision, JaccardIndex
from torch.nn import  NLLLoss


import numpy as np
import glob
import random

import sys
from PIL import Image
import random
import os
import glob

import pandas as pd



gdf = pd.read_csv('/content/gdrive/MyDrive/Non_Trads_FOI/data/full_svi/gdf.csv')
gdf.drop(columns=['Unnamed: 0'], inplace=True)


class StreetViewData(Dataset):
    """
    Module for flexi data stored in flat structure with labels stored in gdf dataframe 
    """
    def __init__(self, root_dir: str, transform: Optional[transforms.Compose] = None):
        
        self.root_dir = root_dir
        self.transform = transform

        groupin_col = 'BRE_mapping_matches'
        groupin_col = 'BRE_Class'
        grouped = gdf.groupby(['Latitude', 'Longitude', 'latlong_id'])[groupin_col].agg(lambda x:x.value_counts().index[0]).reset_index()
        top_classes = ['Wimpey No-Fines',
                        'Reema Conclad',
                        'BISF Type A1',
                        'Mowlem',
                        'Timber Frame (UK)',
                        'Orlit Type I',
                        'Easiform Type I',
                        'Wates',
                        'Weir No-Fines',
                        'Blackburn',
                        'Cornish Unit Type I',
                        'Airey',
                        'Concrete   ',
                        'Bison Trimline',
                        'Unity Type I',
                        'Shepherd',
                        'EDLO BRS',
                        'Parkinson',
                        'Aberdeen Corporation',
                        'Belfry']

        top_classes =['PCC', 'ISC', 'TIM', 'MET']
        
        
        self.classes = top_classes # grouped[groupin_col].unique().tolist()
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.idx_to_classes = {str(i): cls_name for i, cls_name in enumerate(self.classes)}
        print(self.classes)
        
        
        self.samples = []
        list_samples = os.listdir(self.root_dir) 
        ids = [x.split('_')[0] for x in list_samples] 
        labels = [ grouped[grouped['latlong_id'] == x][groupin_col].values  for x in ids ]

        for s, label in zip(list_samples, labels) :
 
            if label in top_classes:
                sample_path = os.path.join(self.root_dir, s)
                if not os.path.exists(sample_path):
                    None
                else:
                    self.samples.append((sample_path, self.class_to_idx[label[0]]))
            

    def classes_dict(self):
        return self.idx_to_classes

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample_path, label = self.samples[index]

        with open(sample_path, "rb") as f:
            sample = Image.open(f).convert("RGB")
            left = 170
            top = 250
            right = 430
            bottom = 10
            sample = sample.crop((left, bottom, right, top))

        if self.transform:
            sample = self.transform(sample)

        return sample, label

class SVI_module(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, transform=None, num_workers=4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = T.Compose([ T.Resize( size = (224, 224) ), T.ToTensor() ])       
        self.dataset = StreetViewData(data_dir, self.transform)

       
        

    def setup(self, stage=None): 
        # split into train, val, and test sets
        num_samples = len(self.dataset)
        indices = list(range(num_samples))
        split_train = int(np.floor(0.7 * num_samples))
        split_val = int(np.floor(0.15 * num_samples))
        np.random.shuffle(indices)
        train_indices, val_indices, test_indices = indices[:split_train], indices[split_train:split_train+split_val], indices[split_train+split_val:]
        print('Num train:{} Num val:{} Num test:{}'.format(len(train_indices), len(val_indices), len(test_indices)) )
        print('Total num: {}  total sum: {}'.format(num_samples, (len(train_indices)+ len(val_indices)+ len(test_indices))))
        # define samplers
        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)
        test_sampler = SubsetRandomSampler(test_indices)

        
        # set up data loaders
        self.train_loader = DataLoader(self.dataset, batch_size=self.batch_size,
                                        sampler=train_sampler, num_workers=self.num_workers)
        self.val_loader = DataLoader(self.dataset, batch_size=self.batch_size,
                                        sampler=val_sampler, num_workers=self.num_workers)
        self.test_loader = DataLoader(self.dataset, batch_size=self.batch_size,
                                      sampler=test_sampler, num_workers=self.num_workers)
    
    

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader



class TransferLearning(pl.LightningModule):
    """
        Transfer learning data modules, works for ResNet Architectures 
        model: pytorch Resnet18 / 50 etc 
    """

    def __init__(self, model, learning_rate, optimiser = 'Adam', class_names=None, num_classes = None):
        super().__init__()
        
        self.optimiser = optimiser
        self.thresh  =  0.5
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.class_names = class_names
        self.num_classes = len(self.class_names)
        
        #add metrics for tracking 
        self.accuracy = Accuracy(task= 'multiclass', num_classes=self.num_classes)
        self.loss= nn.CrossEntropyLoss()
        self.recall = Recall(task= 'multiclass', num_classes=self.num_classes, threshold=self.thresh, average ='macro')
        self.prec = Precision( task= 'multiclass', num_classes=self.num_classes, average='macro')
        self.jacq_ind = JaccardIndex(task= 'multiclass', num_classes=self.num_classes)
        self.confmat = ConfusionMatrix(task="multiclass", num_classes=self.num_classes)


        # init model
        backbone = model
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)

        # use the pretrained model to classify damage 2 classes
        
        self.classifier = nn.Linear(num_filters, self.num_classes)

    def forward(self, x):
        self.feature_extractor.eval()
        with torch.no_grad():
            representations = self.feature_extractor(x).flatten(1)
        x = self.classifier(representations)
        return x
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        
        # training metrics
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        recall = self.recall(preds, y)
        precision = self.prec(preds, y)
        jac = self.jacq_ind(preds, y)
        

        wandb.log({"train_loss": loss})   
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, logger=True)
        self.log('train_recall', recall, on_step=True, on_epoch=True, logger=True)
        self.log('train_precision', precision, on_step=True, on_epoch=True, logger=True)
        self.log('train_jacc', jac, on_step=True, on_epoch=True, logger=True)
        
        return loss
  
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)

        # validation metrics
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        recall = self.recall(preds, y)
        precision = self.prec(preds, y)
        jac = self.jacq_ind(preds, y)
        self.confmat.update(preds, y)
        print(acc, recall, precision )
        

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        self.log('val_recall', recall, prog_bar=True)
        self.log('val_precision', precision, prog_bar=True)
        self.log('val_jacc', jac, prog_bar=True)
        
        return  loss
      

    def on_validation_epoch_end(self):
        confmat = self.confmat.compute()
        class_names = self.class_names
        num_classes = len(class_names)

        df_cm = pd.DataFrame(confmat.cpu().numpy() , index = [i for i in class_names], columns = [i for i in class_names])
        df_cm.to_csv('raw_nums.csv') # used this to validate the number of val samples
        print('Num of val samples: {}. Check this aligns with the numbers from the dataloader'.format(df_cm.sum(axis=1).sum() ))
        #normalise the confusion matrix 
        norm =  np.sum(df_cm, axis=1)
        normalized_cm = (df_cm.T/norm).T
        #validate the confusion matrix sums to num of classes
       
          

        normalized_cm.to_csv('norm_cdf.csv') 
        #log to wandb
        f, ax = plt.subplots(figsize = (15,10)) 
        sn.heatmap(normalized_cm, annot=True, ax=ax)
        wandb.log({"Validation Confusion Matrix ": wandb.Image(f) })
        self.confmat.reset()  #This was NEEDED otherwise the confusion matrix kept stacking the results

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        
        # validation metrics
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        recall = self.recall(preds, y)
        precision = self.prec(preds, y)
        jac = self.jacq_ind(preds, y)
        # confmat = self.confmat(preds, y)

        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', acc, prog_bar=True)
        self.log('test_recall', recall, prog_bar=True)
        self.log('test_precision', precision, prog_bar=True)
        self.log('test_jacc', jac, prog_bar=True)
         
        # self.log('test_conf', confmat, on_step=False, on_epoch=True, logger=True)

        return loss
    def configure_optimizers(self,):
        print('Optimise with {}'.format(self.optimiser) )
        # optimizer = self.optimiser_dict[self.optimiser](self.parameters(), lr=self.learning_rate)
                
                # Support Adam, SGD, RMSPRop and Adagrad as optimizers.
        if self.optimiser == "Adam":
            optimiser = optim.AdamW(self.parameters(), lr = self.learning_rate)
        elif self.optimiser == "SGD":
            optimiser = optim.SGD(self.parameters(), lr = self.learning_rate)
        elif self.optimiser == "Adagrad":
            optimiser = optim.Adagrad(self.parameters(), lr = self.learning_rate)
        elif self.optimiser == "RMSProp":
            optimiser = optim.RMSprop(self.parameters(), lr = self.learning_rate)
        else:
            assert False, f"Unknown optimizer: \"{self.optimiser}\""

        return optimiser

    
    
    
    


class SVI_Resnet(pl.LightningModule):
    """
        Transfer learning data modules, works for ResNet Architectures 
        model: pytorch Resnet18 / 50 etc 
    """

    def __init__(self, model, learning_rate, optimiser = 'Adam', class_names=None, num_classes = None):
        super().__init__()
        
        self.optimiser = optimiser
        self.thresh  =  0.5
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.class_names = class_names
        self.num_classes = len(self.class_names)
        
        #add metrics for tracking 
        self.accuracy = Accuracy(task= 'multiclass', num_classes=self.num_classes)
        self.loss= nn.CrossEntropyLoss()
        self.recall = Recall(task= 'multiclass', num_classes=self.num_classes, threshold=self.thresh, average ='macro')
        self.prec = Precision( task= 'multiclass', num_classes=self.num_classes, average='macro')
        self.jacq_ind = JaccardIndex(task= 'multiclass', num_classes=self.num_classes)
        self.confmat = ConfusionMatrix(task="multiclass", num_classes=self.num_classes)


        # init model
        backbone = model
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)

        # use the pretrained model to classify damage 2 classes
        
        self.classifier = nn.Linear(num_filters, self.num_classes)

    def forward(self, x):
        self.feature_extractor.eval()
        with torch.no_grad():
            representations = self.feature_extractor(x).flatten(1)
        x = self.classifier(representations)
        return x
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        
        # training metrics
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        recall = self.recall(preds, y)
        precision = self.prec(preds, y)
        jac = self.jacq_ind(preds, y)
        

        wandb.log({"train_loss": loss})   
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, logger=True)
        self.log('train_recall', recall, on_step=True, on_epoch=True, logger=True)
        self.log('train_precision', precision, on_step=True, on_epoch=True, logger=True)
        self.log('train_jacc', jac, on_step=True, on_epoch=True, logger=True)
        
        return loss
  
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)

        # validation metrics
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        recall = self.recall(preds, y)
        precision = self.prec(preds, y)
        jac = self.jacq_ind(preds, y)
        self.confmat.update(preds, y)
        print(acc, recall, precision )
        

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        self.log('val_recall', recall, prog_bar=True)
        self.log('val_precision', precision, prog_bar=True)
        self.log('val_jacc', jac, prog_bar=True)
        
        return  loss
      

    def on_validation_epoch_end(self):
        confmat = self.confmat.compute()
        class_names = self.class_names
        num_classes = len(class_names)

        df_cm = pd.DataFrame(confmat.cpu().numpy() , index = [i for i in class_names], columns = [i for i in class_names])
        df_cm.to_csv('raw_nums.csv') # used this to validate the number of val samples
        print('Num of val samples: {}. Check this aligns with the numbers from the dataloader'.format(df_cm.sum(axis=1).sum() ))
        #normalise the confusion matrix 
        norm =  np.sum(df_cm, axis=1)
        normalized_cm = (df_cm.T/norm).T
        #validate the confusion matrix sums to num of classes
       
          

        normalized_cm.to_csv('norm_cdf.csv') 
        #log to wandb
        f, ax = plt.subplots(figsize = (15,10)) 
        sn.heatmap(normalized_cm, annot=True, ax=ax)
        wandb.log({"Validation Confusion Matrix ": wandb.Image(f) })
        self.confmat.reset()  #This was NEEDED otherwise the confusion matrix kept stacking the results

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        
        # validation metrics
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        recall = self.recall(preds, y)
        precision = self.prec(preds, y)
        jac = self.jacq_ind(preds, y)
        # confmat = self.confmat(preds, y)

        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', acc, prog_bar=True)
        self.log('test_recall', recall, prog_bar=True)
        self.log('test_precision', precision, prog_bar=True)
        self.log('test_jacc', jac, prog_bar=True)
         
        # self.log('test_conf', confmat, on_step=False, on_epoch=True, logger=True)

        return loss
    def configure_optimizers(self,):
        print('Optimise with {}'.format(self.optimiser) )
        # optimizer = self.optimiser_dict[self.optimiser](self.parameters(), lr=self.learning_rate)
                
                # Support Adam, SGD, RMSPRop and Adagrad as optimizers.
        if self.optimiser == "Adam":
            optimiser = optim.AdamW(self.parameters(), lr = self.learning_rate)
        elif self.optimiser == "SGD":
            optimiser = optim.SGD(self.parameters(), lr = self.learning_rate)
        elif self.optimiser == "Adagrad":
            optimiser = optim.Adagrad(self.parameters(), lr = self.learning_rate)
        elif self.optimiser == "RMSProp":
            optimiser = optim.RMSprop(self.parameters(), lr = self.learning_rate)
        else:
            assert False, f"Unknown optimizer: \"{self.optimiser}\""

        return optimiser
