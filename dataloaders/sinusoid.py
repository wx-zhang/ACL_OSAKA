import torch
import os
import numpy as np 
from dataloaders.baseloader import *

def init_dataloaders(args):


    '''under construction'''
    from  Data.harmonics import Harmonics
    args.is_classification_task = False



    def make_dataset(mode,train: bool = True) -> torch.Tensor:
        return torch.from_numpy(
            Harmonics(mode,train=train).data
        ).float()

    dataset = make_dataset(0)[0]
    args.input_size = [1,Harmonics(0).data.shape[-1]-1,1]
    meta_dataset = dataset[:2500]
    meta_train = meta_dataset[:2000]
    meta_val  = meta_dataset[2000:]
    meta_test = dataset[2500:]



    cl_dataset = make_dataset(0,train=False)
    cl_ood_dataset1 = make_dataset(1,train=False)
    cl_ood_dataset2 = make_dataset(2,train=False)



    args.prob_pretrain, args.prob_ood1, args.prob_ood2 = 0.5, 0.1, 0.4

    

    meta_train_dataset = MetaDataset(meta_train , args=args)
    meta_val_dataset = MetaDataset(meta_val, args=args)
    meta_test_dataset = MetaDataset(meta_test, args=args)

    meta_train_dataloader = torch.utils.data.DataLoader(meta_train_dataset,
            batch_size=args.batch_size)
    meta_val_dataloader = torch.utils.data.DataLoader(meta_val_dataset,
            batch_size=args.batch_size)
    meta_test_dataloader = torch.utils.data.DataLoader(meta_test_dataset,
            batch_size=args.batch_size)

    meta_dataloader = {}
    meta_dataloader[0] = {}
    meta_dataloader[0]['train'] = meta_train_dataloader
    meta_dataloader[0]['valid'] = meta_val_dataloader
    meta_dataloader[0]['test'] = meta_test_dataloader
    meta_dataloader[0]['name'] = f'sinusoid meta set'

    
    cl_dataloader = StreamDataset(cl_dataset, cl_ood_dataset1, cl_ood_dataset2,args=args)
    cl_dataloader = torch.utils.data.DataLoader(cl_dataloader, batch_size=1)

    del meta_dataset, meta_train_dataset, meta_val_dataset, meta_test_dataset,\
     cl_dataset, cl_ood_dataset1, cl_ood_dataset2,\
     meta_train_dataloader, meta_val_dataloader, meta_test_dataloader

    return meta_dataloader, cl_dataloader



