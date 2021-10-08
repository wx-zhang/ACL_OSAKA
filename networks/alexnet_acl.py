# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import utils

class Shared(torch.nn.Module):

    def __init__(self,args):
        super(Shared, self).__init__()

        self.nchannel,self.size1,self.size2 = args.input_size
        self.num_ways=args.num_ways
        self.hidden_size = args.hidden_size

        if args.dataset == 'tiered_imagenet':
            hiddens = [64, 128, 256, 512, 512, 512]
        else:
            raise NotImplementedError

        size = self.size1
        self.conv1=torch.nn.Conv2d(self.nchannel,hiddens[0],kernel_size=size//8)
        s=utils.compute_conv_output_size(size,size//8)
        s=s//2
        self.conv2=torch.nn.Conv2d(hiddens[0],hiddens[1],kernel_size=size//10)
        s=utils.compute_conv_output_size(s,size//10)
        s=s//2
        self.conv3=torch.nn.Conv2d(hiddens[1],hiddens[2],kernel_size=2)
        s=utils.compute_conv_output_size(s,2)
        s=s//2
        self.maxpool=torch.nn.MaxPool2d(2)
        self.relu=torch.nn.ReLU()

        self.drop1=torch.nn.Dropout(0.2)
        self.drop2=torch.nn.Dropout(0.5)
        self.fc1=torch.nn.Linear(hiddens[2]*s*s,hiddens[3])
        self.fc2=torch.nn.Linear(hiddens[3],hiddens[4])
        self.fc3=torch.nn.Linear(hiddens[4],hiddens[5])
        self.fc4=torch.nn.Linear(hiddens[5], self.hidden_size)


    def forward(self, x_s):
        x_s = x_s.view_as(x_s)
        h = self.maxpool(self.drop1(self.relu(self.conv1(x_s))))
        h = self.maxpool(self.drop1(self.relu(self.conv2(h))))
        h = self.maxpool(self.drop2(self.relu(self.conv3(h))))
        h = h.view(x_s.size(0), -1)
        h = self.drop2(self.relu(self.fc1(h)))
        h = self.drop2(self.relu(self.fc2(h)))
        h = self.drop2(self.relu(self.fc3(h)))
        h = self.drop2(self.relu(self.fc4(h)))
        return h



class Private(torch.nn.Module):
    def __init__(self, args):
        super(Private, self).__init__()

        self.nchannel,self.size,_=args.input_size
        self.num_ways=args.num_ways
        self.hidden_size = args.hidden_size
        self.num_domains = args.num_domains
        self.device = args.device

        if args.dataset ==  'tiered_imagenet':
            hiddens=[16,16]
            flatten=3600
        else:
            raise NotImplementedError


        self.task_out = torch.nn.ModuleList()
        for _ in range(self.num_tasks):
            self.conv = torch.nn.Sequential()
            self.conv.add_module('conv1',torch.nn.Conv2d(self.nchannel, hiddens[0], kernel_size=self.size // 8))
            self.conv.add_module('relu1', torch.nn.ReLU(inplace=True))
            self.conv.add_module('drop1', torch.nn.Dropout(0.2))
            self.conv.add_module('maxpool1', torch.nn.MaxPool2d(2))
            self.conv.add_module('conv2', torch.nn.Conv2d(hiddens[0], hiddens[1], kernel_size=self.size // 10))
            self.conv.add_module('relu2', torch.nn.ReLU(inplace=True))
            self.conv.add_module('dropout2', torch.nn.Dropout(0.5))
            self.conv.add_module('maxpool2', torch.nn.MaxPool2d(2))
            self.task_out.append(self.conv)
            self.linear = torch.nn.Sequential()

            self.linear.add_module('linear1', torch.nn.Linear(flatten,self.hidden_size))
            self.linear.add_module('relu3', torch.nn.ReLU(inplace=True))
            self.task_out.append(self.linear)


    def forward(self, x, task_id):
        x = x.view_as(x)
        out = self.task_out[2*task_id].forward(x)
        out = out.view(out.size(0),-1)
        out = self.task_out[2*task_id+1].forward(out)
        return out



class Net(torch.nn.Module):

    def __init__(self, args):
        super(Net, self).__init__()
        self.nchannel,size,_=args.input_size
        self.num_ways=args.num_ways
        self.hidden_size = args.hidden_size
        self.num_domains = args.num_domains
        self.image_size = self.nchannel*size*size
        self.args=args

        self.hidden1 = args.head_units
        self.hidden2 = args.head_units//2

        self.shared = Shared(args)
        self.private = Private(args)

        self.head = torch.nn.ModuleList()
        for i in range(self.num_domains):
            self.head.append(
                torch.nn.Sequential(
                    torch.nn.Linear(2*self.hidden_size, self.hidden1),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Dropout(),
                    torch.nn.Linear(self.hidden1, self.hidden2),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Linear(self.hidden2, args.num_ways)
                ))


    def forward(self, x_s, x_p, tt, task_id):

        x_s = x_s.view_as(x_s)
        x_p = x_p.view_as(x_p)

        x_s = self.shared(x_s)
        x_p = self.private(x_p, task_id)

        x = torch.cat([x_p, x_s], dim=1)


        return torch.stack([self.head[tt[i]].forward(x[i]) for i in range(x.size(0))])


    def get_encoded_ftrs(self, x_s, x_p, task_id):
        return self.shared(x_s), self.private(x_p, task_id)

    def print_model_size(self,wandb=None):
        count_P = sum(p.numel() for p in self.private.parameters() if p.requires_grad)
        count_S = sum(p.numel() for p in self.shared.parameters() if p.requires_grad)
        count_H = sum(p.numel() for p in self.head.parameters() if p.requires_grad)

        print('Num parameters in Shared       = %s ' % (self.pretty_print(count_S)))
        print('Num parameters in Private      = %s,  per task = %s ' % (self.pretty_print(count_P),self.pretty_print(count_P/self.num_domains)))
        print('Num parameters in Header       = %s,  per task = %s ' % (self.pretty_print(count_H),self.pretty_print(count_H/self.num_domains)))
        print('Num parameters in P+H          = %s ' % self.pretty_print(count_P+count_H))
        print('-------------------------->   Total architecture size: %s parameters (%sB)' % (self.pretty_print(count_S + count_P + count_H),
                                                                    self.pretty_print(4*(count_S + count_P + count_H))))
        if wandb:
            wandb.log({'Num parameters in Shared':count_S})
            wandb.log({'Num parameters in P':count_P,'Num parameters in P per task':count_P/self.num_domains})
            wandb.log({'Num parameters in Private':count_H,'Num parameters in P per task':count_H/self.num_domains})
            wandb.log({'Num parameters in P+H':count_P+count_H})
            wandb.log({'Total architecture size':count_S + count_P + count_H,'Total parameter size':4*(count_S + count_P + count_H)})

    def pretty_print(self, num):
        magnitude=0
        while abs(num) >= 1000:
            magnitude+=1
            num/=1000.0
        return '%.1f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])

        
    def freeze_s_module(self, freeze=True):
        for param in self.shared.parameters():
            param.requires_grad = not freeze
            if freeze:
                param.grad = None
