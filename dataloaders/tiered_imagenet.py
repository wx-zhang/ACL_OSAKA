import sys
import torch
import numpy as np
from pdb import set_trace
import os
from dataloaders.baseloader import *


def init_dataloaders(args):



    from Data.tiered_imagenet import NonEpisodicTieredImagenet

    args.prob_pretrain, args.prob_ood1, args.prob_ood2 = 0.25, 0.25, 0.5

    args.is_classification_task = True
    args.n_train_cls = 100
    args.n_val_cls = 100
    args.n_train_samples = 500


    args.input_size = [3,84,84]
    tiered_dataset = NonEpisodicTieredImagenet(args.folder, split="train")



    meta_train = {}
    meta_val = {}
    meta_test = {}
    cl_ood_dataset1 = {}


    cls_cnt = 0
    for k in tiered_dataset.data.keys():
        if cls_cnt > args.n_train_cls + args.n_val_cls :
            cl_ood_dataset1[k] = tiered_dataset.data[k]
        elif cls_cnt > args.n_train_cls:
            meta_test[k] = tiered_dataset.data[k]
            cls_cnt += len(meta_test[k])
        else:
            meta_train[k] = tiered_dataset.data[k][:,:args.n_train_samples, ...]
            meta_val[k] = tiered_dataset.data[k][:,args.n_train_samples:,...]
            cls_cnt += len(meta_train[k])
        


    meta_train_dataset =  MultiDomainMetaDataset(meta_train , args=args)
    meta_val_dataset =  MultiDomainMetaDataset(meta_val, train=False,args=args)
    meta_test_dataset = MultiDomainMetaDataset(meta_test, train=False,args=args)
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
    meta_dataloader[0]['name'] = f'tiered-imagenet meta set'

    cl_dataset = tiered_dataset.data
    cl_ood_dataset2 = NonEpisodicTieredImagenet(args.folder, split="val").data
    args.num_domains = len(cl_dataset.keys())
    for k in cl_ood_dataset2.keys():
        cl_ood_dataset2[k+args.num_domains] = cl_ood_dataset2.pop(k)
    args.num_domains += len(cl_ood_dataset2)

    cl_ood_dataset1 = cl_ood_dataset1
    cl_ood_dataset2 = cl_ood_dataset2
    cl_dataloader = MultiDomainStreamDataset(cl_dataset, cl_ood_dataset1, cl_ood_dataset2,args=args)
    cl_dataloader = torch.utils.data.DataLoader(cl_dataloader, batch_size=1)

 

    del tiered_dataset, meta_train_dataset, meta_val_dataset, meta_test_dataset,\
     cl_dataset, cl_ood_dataset1, cl_ood_dataset2,\
     meta_train_dataloader, meta_val_dataloader, meta_test_dataloader

    return meta_dataloader, cl_dataloader



