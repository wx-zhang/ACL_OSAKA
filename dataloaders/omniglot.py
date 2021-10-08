import sys
import torch
import numpy as np
from pdb import set_trace
import os
from dataloaders.baseloader import *

def init_dataloaders(args):


    from Data.omniglot import Omniglot
    from torchvision.datasets import MNIST, FashionMNIST

    args.is_classification_task = True
    args.prob_pretrain, args.prob_ood1, args.prob_ood2 = 0.5, 0.25, 0.25
    args.n_train_cls = 100
    args.n_val_cls = 10
    args.n_train_samples = args.num_shots


    args.input_size = [1,28,28]
    Omniglot_dataset = Omniglot(args.folder).data
    Omniglot_dataset = torch.from_numpy(Omniglot_dataset).type(torch.float).to(args.device)
    meta_train_dataset = Omniglot_dataset[:args.n_train_cls]
    meta_train = meta_train_dataset[:,:args.n_train_samples,:,:]
    meta_val = meta_train_dataset[:,args.n_train_samples:,:,:]

    meta_test = Omniglot_dataset[args.n_train_cls : (args.n_train_cls+args.n_val_cls)]


    cl_dataset = Omniglot_dataset
    cl_ood_dataset1 = MNIST(args.folder, train=True,  download=True)
    cl_ood_dataset2 = FashionMNIST(args.folder, train=True,  download=True)
    cl_ood_dataset1, _ = order_and_split(cl_ood_dataset1.data, cl_ood_dataset1.targets)
    cl_ood_dataset2, _ = order_and_split(cl_ood_dataset2.data, cl_ood_dataset2.targets)
    cl_ood_dataset1 = cl_ood_dataset1[:,:,None,:,:]
    cl_ood_dataset2 = cl_ood_dataset2[:,:,None,:,:]
    cl_ood_dataset1 = cl_ood_dataset1.type(torch.float).to(args.device)
    cl_ood_dataset2 = cl_ood_dataset2.type(torch.float).to(args.device)


  
    meta_train_dataset = MetaDataset(meta_train , args=args)
    meta_val_dataset = MetaDataset(meta_val, train=False,args=args)
    meta_test_dataset = MetaDataset(meta_test, train=False,args=args)

    meta_train_dataloader = torch.utils.data.DataLoader(meta_train_dataset,
            batch_size=args.batch_size,drop_last=True)
    meta_val_dataloader = torch.utils.data.DataLoader(meta_val_dataset,
            batch_size=args.batch_size,drop_last=True)
    meta_test_dataloader = torch.utils.data.DataLoader(meta_test_dataset,
            batch_size=args.batch_size,drop_last=True)
    meta_dataloader = {}
    meta_dataloader[0] = {}
    meta_dataloader[0]['train'] = meta_train_dataloader
    meta_dataloader[0]['valid'] = meta_val_dataloader
    meta_dataloader[0]['test'] = meta_test_dataloader
    meta_dataloader[0]['name'] = f'omniglot meta set'

    
    cl_dataloader = StreamDataset(cl_dataset, cl_ood_dataset1, cl_ood_dataset2,args=args)
    cl_dataloader = torch.utils.data.DataLoader(cl_dataloader, batch_size=1)

    del Omniglot_dataset, meta_train_dataset, meta_val_dataset, meta_test_dataset,\
     cl_dataset, cl_ood_dataset1, cl_ood_dataset2,\
     meta_train_dataloader, meta_val_dataloader, meta_test_dataloader

    return meta_dataloader, cl_dataloader

