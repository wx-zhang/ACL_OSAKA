from PIL import Image
from torch.utils.data import Dataset
import os
import torch
import pickle as pkl
#import cPickle as pkl
from io import BytesIO
from torchvision import transforms as t
import numpy as np
from pdb import set_trace

class NonEpisodicTieredImagenet(Dataset):
    tasks_type = "clss"
    name = "tiered-imagenet"
    split_paths = {"train":"train", "test":"test", "valid": "val", "val": "val"}
    c = 3
    h = 84
    w = 84


    def __init__(self, path, split, transforms=t.ToTensor(), **kwargs):
        """ Constructor

        Args:
            split: data split
            few_shot_sampler: FewShotSampler instance
            task: dataset task (if more than one)
            size: number of tasks to generate (int)
            disjoint: whether to create disjoint splits.
        """
        split = self.split_paths[split]
        self.ROOT_PATH = path


        if not os.path.exists(self.ROOT_PATH+'/{}-tiered-imagenet-acl.npy'.format(split)): #'/tmp/{}-tiered-imagenet.npy'.format(split)):
            #FIXME: change me back
            # if not os.path.exists(self.ROOT_PATH):
            #     print(
            #         "Please download tiered-imagenet as indicated in https://github.com/renmengye/few-shot-ssl-public")
            #     raise IOError
            
            _ROOT_PATH = os.path.join(path,'tiered-imagenet' )
            img_path = os.path.join(_ROOT_PATH, "%s_images_png.pkl" % (split))
            label_path = os.path.join(_ROOT_PATH, "%s_labels.pkl" % (split))
            self.transforms = transforms
            with open(img_path, 'rb') as infile:
                images = pkl.load(infile, encoding="bytes")

            with open(label_path, 'rb') as infile:
                self.labels = pkl.load(infile, encoding="bytes")
                self.labels_specific = self.labels["label_specific"]
                self.labels_general = self.labels["label_general"]

            print("Loading tiered-imagenet...")
            label_count = {i: (self.labels_specific == i).astype(int).sum() for i in set(self.labels_specific)}
            cat_count = {i: len(np.unique(self.labels_specific[self.labels_general==i])) for i in set(self.labels_general)}
            print (f'number of classes for every categories: {cat_count}')



            min_count = np.min(list(label_count.values()))
            print (f'{min_count} images for every classes')


            self.data = {}


            label_count = {i: 0 for i in set(self.labels_specific)}
            cat_label_count = {i:[]  for i in set(self.labels_general)}
            for im, label, cat in zip(images, self.labels_specific,self.labels_general):
                if cat not in self.data.keys():
                    self.data[cat] = torch.zeros( cat_count[cat], min_count, self.c, self.h, self.w, dtype=torch.uint8)
                if label not in cat_label_count[cat]:
                    cat_label_count[cat].append(label)

                index = label_count[label]
                if index == min_count:
                    continue
                else:
                    cls_idx = cat_label_count[cat].index(label)
                    self.data[cat][cls_idx, index, ...] = torch.from_numpy(np.transpose(self.__decode(im).resize((self.h, self.w),Image.NEAREST), [2,0,1]))
                    label_count[label] += 1
            np.save(os.path.join(self.ROOT_PATH,'%s-tiered-imagenet-acl' % (split)), self.data,allow_pickle=True)
            del (images)

        else:
            self.data = np.load(os.path.join(self.ROOT_PATH, '{}-tiered-imagenet-acl.npy'.format(split)),
                                allow_pickle=True).item()
        print(f"Prepare {split} Done")

    # def from_hierarchy(self, start, end):
    #     ret = []
    #     mask = (self.labels_general >= start) * (self.labels_general < end)
    #     specific_set = np.unique(self.labels_specific[mask])
    #     return self.data[:, specific_set, ...]

    def __decode(self, image):
        return Image.open(BytesIO(image)).convert("RGB")

    def __getitem__(self, item):
        return self.transforms(self.__decode(self.images[item])), self.labels[item]

    def __len__(self):
        return self.data.shape[0] * self.data.shape[1]
