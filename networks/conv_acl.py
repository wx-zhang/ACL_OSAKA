# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch


class Private(torch.nn.Module):
    # single private module
    def __init__(self, args):
        super(Private, self).__init__()

        self.nchannel,self.size1,self.size2 = args.input_size
        self.num_ways=args.num_ways
        self.hidden_size = args.hidden_size
        self.device = args.device
        self.num_domains = args.num_domains

        self.task_out = torch.nn.ModuleList()
        for _ in range(self.num_domains):
            self.conv = torch.nn.Sequential()
            self.conv.add_module('conv2', torch.nn.Conv2d(self.nchannel, self.hidden_size,kernel_size=3,
                                  stride=1, padding=1, bias=True))
            #self.linear.add_module('relu', torch.nn.ReLU(inplace=True))
            self.conv.add_module('leakyrelu', torch.nn.LeakyReLU(0.001))
            self.conv.add_module('maxpool',torch.nn.MaxPool2d(2))
            self.conv.add_module('linear',torch.nn.Linear(self.nchannel*self.size1*self.size2,self.hidden_size))
            self.task_out.append(self.conv)



        

    def forward(self, x_p, domain_id):
        x = self.task_out[domain_id].forward(x_p)

        return 



class Shared(torch.nn.Module):

    def __init__(self,args):
        super(Shared, self).__init__()


        self.nchannel,self.size1,self.size2 = args.input_size

        self.num_ways=args.num_ways
        self.hidden_size = args.hidden_size
        self.s_units = args.s_units

        self.nlayers = args.s_layers

        self.relu=torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(2)
        self.drop=torch.nn.Dropout(0.2)
        self.fc_input=torch.nn.Conv2d(self.nchannel, self.s_units, kernel_size=3,stride=1, padding=1,bias=True)
        self.fc_hidden = torch.nn.Conv2d(self.s_units, self.s_units, kernel_size=3,stride=1, padding=1,bias=True)
        last_input = self.hidden_size*self.nchannel*self.size1*self.size2/(2*2)
        for i in range(self.nlayers):
            last_input = last_input/4
        last_input = int(last_input)
        self.fc_output=torch.nn.Linear(last_input,self.hidden_size)


    def forward(self, x_s):


        h = self.drop(self.pool(self.relu(self.fc_input(x_s))))

        for i in range(self.nlayers):
            h = self.drop(self.pool(self.relu(self.fc_hidden(h))))

        h = h.view(h.size(0), -1)
        h = self.drop(self.relu(self.fc_output(h)))

        return h


class Net(torch.nn.Module):

    def __init__(self, args):
        super(Net, self).__init__()
        self.nchannel,self.size1,self.size2 = args.input_size
        self.num_ways=args.num_ways
        self.hidden_size = args.hidden_size

        self.device = args.device
        self.slayers = args.s_layers
        

        if args.dataset == 'sinusoid':
            self.hidden1 = 14
            self.hidden2 = 7
        elif args.dataset == 'omniglot': 
            self.hidden1 = 28
            self.hidden2 = 14



        self.shared = Shared(args)
        self.private = Private(args)
        self.num_domains = args.num_domains

        self.bn = torch.nn.BatchNorm1d(2 * self.hidden_size)
        
        self.head = torch.nn.ModuleList()




        for i in range(self.num_domains):
            self.head.append(torch.nn.Sequential(
                    torch.nn.Linear(self.hidden_size*2, self.hidden1),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Dropout(),
                    torch.nn.Linear(self.hidden1, self.hidden2),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Linear(self.hidden2, self.num_ways)
                    ))



    def forward(self,x_s, x_p, tt, domain_id):





        x_s = self.shared(x_s)
        x_p = self.private(x_p,domain_id)
        x_s = x_s.view(x_s.size(0), -1)
        x_p = x_p.view(x_p.size(0), -1)


        x = torch.cat([x_s, x_p], dim=1)
        #x = self.bn(x)
        # print (x_s.shape,x_p.shape,x.shape)
        # print (self.headinput)



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
        return '%.2f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])

    def freeze_s_module(self, freeze=True):
        for param in self.shared.parameters():
            param.requires_grad = not freeze
            if freeze:
                param.grad = None
