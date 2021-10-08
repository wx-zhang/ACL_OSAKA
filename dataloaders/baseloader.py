import sys
import torch
import numpy as np


# --------------------------------------------------------------------------
# utils
# --------------------------------------------------------------------------

def select_from_tensor(tensor, index):
    """ equivalent to tensor[index] but for batched / 2D+ tensors """

    last_dim = index.dim() - 1

    assert tensor.dim() >= index.dim()
    assert index.size()[:last_dim] == tensor.size()[:last_dim]

    # we have to make `train_idx` the same shape as train_data, or else
    # `torch.gather` complains.
    # see https://discuss.pytorch.org/t/batched-index-select/9115/5

    missing_dims = tensor.dim() - index.dim()
    index = index.view(index.size() + missing_dims * (1,))
    index = index.expand((-1,) * (index.dim() - missing_dims) + tensor.size()[(last_dim+1):])

    return torch.gather(tensor, last_dim, index)

def order_and_split(data_x, data_y):
    """ given a dataset, returns (num_classes, samples_per_class, *data_x[0].size())
        tensor where samples (and labels) are ordered and split per class """

    # sort according to the label
    out_train = [
        (x,y) for (x,y) in sorted(zip(data_x, data_y), key=lambda v : v[1]) ]

    # stack in increasing label order
    data_x, data_y = [
            torch.stack([elem[i] for elem in out_train]) for i in [0,1] ]
    

    # find first indices of every class
    n_classes = data_y.unique().size(0)
    idx       = [((data_y + i) % n_classes).argmax() for i in range(n_classes)]+[data_y.shape[0]]

    idx       =  [x + 1 for x in sorted(idx)]

    # split into different classes
    to_chunk = [a - b for (a,b) in zip(idx[1:], idx[:-1])]

    data_x   = data_x.split(to_chunk)
    data_y   = data_y.split(to_chunk)

    # give equal amt of points for every class
    #TODO(if this is restrictive for some dataset, we can change)
    min_amt  = min([x.size(0) for x in data_x])
    data_x   = torch.stack([x[:min_amt] for x in data_x])
    data_y   = torch.stack([y[:min_amt] for y in data_y])

    # sanity check
    for i, item in enumerate(data_y):
        assert item.unique().size(0) == 1 and item[0] == i, 'wrong result'

    return data_x, data_y








# --------------------------------------------------------------------------
# Datasets and Streams (the good stuff)
# --------------------------------------------------------------------------

class MetaDataset(torch.utils.data.Dataset):
    """ Dataset similar to BatchMetaDataset in TorchMeta """

    def __init__(self, data, train=True, args=None, **kwargs):

        '''
        Parameters
        ----------

        data : Array of (x,) pairs, one for each class. Contains all the
            training data that should be available at meta-training time (inner loop).

        '''


        # separate the classes into tasks
        n_classes   = len(data)

        self._len        = None
        self.n_way       = args.num_ways
        self.kwargs      = kwargs
        self.n_classes   = n_classes
        if train:
            self.n_shots     = args.num_shots
        else:
            self.n_shots     = args.num_shots_test




        self.input_size  = args.input_size
        self.x_dim       = self.input_size[1]
        self.device      = args.device
        self.is_classification_task = args.is_classification_task

        self.all_classes = np.arange(n_classes)

        self.data        = data

    def __len__(self):
        return self.data.size(0)


    def __getitem__(self, index):
        if self.is_classification_task:
            return self._getitem_classification(index)
        else:
            return self._getitem_regression(index)

    def _getitem_regression(self, index):
        # in regression tasks, data = [task, sample point, [x,y]]

        data_t = self.data[index]

        x = data_t[:self.x_dim ]
        y = data_t[self.x_dim: ]



        x = x.to(self.device)
        y = y.to(self.device)
        tt = torch.tensor([0])
        td = torch.tensor(1)
        tt = tt.to(self.device)
        td = td.to(self.device)

        return x,y,tt,td

    def _getitem_classification(self, index):
        # NOTE: This method COMPLETELY ignores the index. This will be a problem
        # if you wish to recover a specific batch of data.

        # classes_in_task = np.random.choice(self.all_classes, self.n_way, replace=False)
        # #print (classes_in_task)
        # samples_in_class = self.data.shape[1]
        # data = self.data[classes_in_task]
        # # sample indices for meta train
        # train_idx = torch.Tensor(self.n_way, self.n_shots)
        # train_idx = train_idx.uniform_(0, samples_in_class).long()
        # train_x = select_from_tensor(data, train_idx)
        # train_x = train_x.view(-1, *self.input_size)

        # # build label tensors
        # train_y = torch.arange(self.n_way).view(-1, 1).expand(-1, self.n_shots)
        # train_y = train_y.flatten()

        # train_x = train_x.float().to(self.device)
        # train_y = train_y.to(self.device)
    
        # # same signature are TorchMeta
        # tt = torch.zeros_like(train_y)
        # td = torch.ones_like(train_y)
        # tt = tt.to(self.device)
        # td = td.to(self.device)
        
        
        data = self.data[index]
        samples_in_class = self.data.shape[1]
        #print (self.data.shape)
        train_idx = torch.Tensor( self.n_shots)
        train_idx = np.random.choice(samples_in_class, self.n_shots, replace=False)
        train_x = data[train_idx].float().to(self.device)
        train_y = (index%self.n_way) * torch.ones(self.n_shots).to(self.device)
        tt = torch.zeros(self.n_shots)
        td = torch.ones(self.n_shots)
        tt = tt.to(self.device)
        td = td.to(self.device)
        
        
        return train_x,train_y,tt,td

class MultiDomainMetaDataset(torch.utils.data.Dataset):
    """ Dataset similar to BatchMetaDataset in TorchMeta """

    def __init__(self, data, train=True, args=None, **kwargs):

        '''
        Only for classification

        Parameters
        ----------

        data : Array of (x,) pairs, one for each class. Contains all the
            training data that should be available at meta-training time (inner loop).

        '''


        # separate the classes into tasks
        n_classes   = len(data)

        self._len        = None
        self.n_way       = args.num_ways
        self.kwargs      = kwargs
        self.n_classes   = n_classes
        if train:
            self.n_shots     = args.num_shots
        else:
            self.n_shots     = args.num_shots_test




        self.input_size  = args.input_size
        self.x_dim       = self.input_size[1]
        self.device      = args.device
        self.is_classification_task = args.is_classification_task

        self.all_classes = np.arange(n_classes)

        self.data        = data
        self.domain      = list(data.keys())
        self.domain_context = [len(self.data[i]) for i in self.domain]
        self.n_domains = len(data.keys())
        self.memory = {}


    def __len__(self):
        nn = 0
        for v in self.data.values():
            nn += len(v)
        return nn


    def __getitem__(self, index):

        domain_count = 0
        domain_index = index
        for i in range(self.n_domains):
            if domain_index < self.domain_context[i]:
                break
            else:
                domain_count += 1
                domain_index -= domain_context[i]

        domain_id = self.domain[domain_count]
        data = self.data[domain_id][domain_index]
        samples_in_class = self.data.shape[1]
        #print (self.data.shape)
        train_idx = torch.Tensor( self.n_shots)
        train_idx = np.random.choice(samples_in_class, self.n_shots, replace=False)
        train_x = data[train_idx].float().to(self.device)
        train_y = (index%self.n_way) * torch.ones(self.n_shots).to(self.device)
        tt = torch.ones(self.n_shots) * domain_id
        td = torch.ones(self.n_shots) * (domain_id+1)
        tt = tt.to(self.device)
        td = td.to(self.device)
        
        
        return train_x,train_y,tt,td


class StreamDataset(torch.utils.data.Dataset):
    """ stream of non stationary dataset as described by Mass """

    def __init__(self, train_data, test_data, ood_data, args=None, **kwargs):

        '''
        Parameters
        ----------

        train_data : Array of (x,) pairs, one for each class. Contains the SAME
            classes used during (meta) training, but different samples.
        test_data  : Array of (x,) pairs, one for each class. These are DIFFERENT
            classes from the ones used during (meta) training.
        n_way      : number of classes per task at cl-test time
        n_shots    : number of samples per classes at cl-test time

        '''
        prob_pretrain = args.prob_pretrain
        prob_ood1     = args.prob_ood1
        prob_ood2     = args.prob_ood2
        

        assert prob_pretrain + prob_ood1 + prob_ood2 == 1.


        self.n_shots    = args.num_shots
        self.n_way      = args.num_ways

        self.modes    = ['pretrain', 'ood1', 'ood2']
        self.modes_id = [0, 1, 2]
        self.probs    = np.array([prob_pretrain, prob_ood1, prob_ood2])
        self.data     = [train_data, test_data, ood_data]
        self.p_statio = args.prob_statio


        self.index_in_task_sequence = 0
        self.steps_done_on_task = 0


        self.input_size  = args.input_size

        self.x_dim       = self.input_size[1]
        self.device      = args.device
        self.is_classification_task = args.is_classification_task
        self.task_sequence = args.task_sequence
        self.n_steps_per_task = args.n_steps_per_task

        self.mode_name_map = dict(zip(self.modes, self.modes_id))

        # starting mode:
        self._mode = np.random.choice([0,1,2], p=self.probs)
        self._classes_in_task = None
        self._samples_in_class = None


    def __len__(self):
        # this is a never ending stream
        return sys.maxsize


    def __getitem__(self, index):
        if self.is_classification_task:
            return self._getitem_classification(index)
        else:
            return self._getitem_regression(index)

    def _getitem_regression(self, index):
        task_switch = False
        if self.task_sequence:
            self.steps_done_on_task += 1

            if self.steps_done_on_task >= self.n_steps_per_task:
                task_switch = True
                self.steps_done_on_task = 0
                self.index_in_task_sequence += 1
                self.index_in_task_sequence %= len(self.task_sequence)

            mode_name = self.task_sequence[self.index_in_task_sequence]
            self._mode = self.mode_name_map[mode_name]
        else:
            if (np.random.uniform() > self.p_statio):
                mode  = np.random.choice(self.modes_id, p=self.probs)
                self._mode = mode
                task_switch = mode != self._mode

        mode_data = self.data[self._mode][index]

        x = mode_data[:, :self.x_dim]

        y = mode_data[:, self.x_dim:]

        x = x.to(self.device)
        y = y.to(self.device)
        tt = torch.ones(len(x)).to(self.device,dtype=torch.int)* self._mode
        td = torch.ones(len(x)).to(self.device)* (self._mode + 1)
        mode = torch.ones(len(x)).to(self.device,dtype=torch.int)* self._mode

        return x, y, tt, td, mode

    def _getitem_classification(self, index):
        # NOTE: This method COMPLETELY ignores the index. This will be a problem
        # if you wish to recover a specific batch of data.

        # NOTE: using multiple workers (`num_workers > 0`) or `batch_size  > 1`
        # will have undefined behaviour. This is because unlike regular datasets
        # here the sampling process is sequential.
        task_switch = 0
        if self.task_sequence:
            self.steps_done_on_task += 1

            if self.steps_done_on_task >= self.n_steps_per_task:
                task_switch = 1
                self.steps_done_on_task = 0
                self.index_in_task_sequence += 1
                self.index_in_task_sequence %= len(self.task_sequence)

            mode_name = self.task_sequence[self.index_in_task_sequence]
            self._mode = self.mode_name_map[mode_name]
        elif (np.random.uniform() > self.p_statio) or (self._classes_in_task is None):
            # mode  = np.random.choice(self.modes_id, p=self.probs)
            # self._mode = mode
            # task_switch = mode != self._mode
            # TODO: this makes a switch even if staying in same mode!
            task_switch = 1
            self._mode  = np.random.choice([0,1,2], p=self.probs)

            mode_data = self.data[self._mode]
            n_classes = len(mode_data)
            self._samples_in_class = mode_data.size(1)

            # sample `n_way` classes
            self._classes_in_task = np.random.choice(np.arange(n_classes), self.n_way,
                    replace=False)

        else:

            task_switch = 0

        mode_data = self.data[self._mode]
        data = mode_data[self._classes_in_task]

        # sample indices for meta train
        idx = torch.Tensor(self.n_way, self.n_shots)#.to(self.device)
        idx = idx.uniform_(0, self._samples_in_class).long()
        if not(self.cpu_dset):
            idx = idx.to(self.device)
        data = select_from_tensor(data, idx)

        # build label tensors
        labels = torch.arange(self.n_way).view(-1, 1).expand(-1, self.n_shots).to(self.device)

        # squeeze
        data = data.view(-1, *self.input_size)
        labels = labels.flatten()

        if self.cpu_dset:
            data = data.float().to(self.device)
            labels = labels.to(self.device)

        return data, labels, task_switch, self.modes[self._mode]



class MultiDomainStreamDataset(torch.utils.data.Dataset):
    """ stream of non stationary dataset as described by Mass """

    def __init__(self, train_data, test_data, ood_data, args=None, **kwargs):

        '''
        Parameters
        ----------

        train_data : Array of (x,) pairs, one for each class. Contains the SAME
            classes used during (meta) training, but different samples.
        test_data  : Array of (x,) pairs, one for eacsh class. These are DIFFERENT
            classes from the ones used during (meta) training.
        n_way      : number of classes per task at cl-test time
        n_shots    : number of samples per classes at cl-test time

        '''
        prob_pretrain = args.prob_pretrain
        prob_ood1     = args.prob_ood1
        prob_ood2     = args.prob_ood2
        

        assert prob_pretrain + prob_ood1 + prob_ood2 == 1.


        self.n_shots    = args.num_shots
        self.n_way      = args.num_ways

        self.modes    = ['pretrain', 'ood1', 'ood2']
        self.modes_id = [0, 1, 2]

        self.probs    = np.array([prob_pretrain, prob_ood1, prob_ood2])
        self.data     = [train_data, test_data, ood_data]
        self.p_statio = args.prob_statio


        self.index_in_task_sequence = 0
        self.steps_done_on_task = 0


        self.input_size  = args.input_size

        self.x_dim       = self.input_size[1]
        self.device      = args.device
        self.is_classification_task = args.is_classification_task
        self.task_sequence = args.task_sequence
        self.n_steps_per_task = args.n_steps_per_task

        self.mode_name_map = dict(zip(self.modes, self.modes_id))

        # starting mode:
        self._mode = np.random.choice([0,1,2], p=self.probs)
        self._classes_in_task = None
        self._samples_in_class = None


    def __len__(self):
        # this is a never ending stream
        return sys.maxsize


    def __getitem__(self, index):

        # NOTE: This method COMPLETELY ignores the index. This will be a problem
        # if you wish to recover a specific batch of data.

        # NOTE: using multiple workers (`num_workers > 0`) or `batch_size  > 1`
        # will have undefined behaviour. This is because unlike regular datasets
        # here the sampling process is sequential.
        task_switch = 0
        if self.task_sequence:
            self.steps_done_on_task += 1

            if self.steps_done_on_task >= self.n_steps_per_task:
                task_switch = 1
                self.steps_done_on_task = 0
                self.index_in_task_sequence += 1
                self.index_in_task_sequence %= len(self.task_sequence)

            mode_name = self.task_sequence[self.index_in_task_sequence]
            self._mode = self.mode_name_map[mode_name]
        elif (np.random.uniform() > self.p_statio) or (self._classes_in_task is None):
            # mode  = np.random.choice(self.modes_id, p=self.probs)
            # self._mode = mode
            # task_switch = mode != self._mode
            # TODO: this makes a switch even if staying in same mode!
            task_switch = 1
            self._mode  = np.random.choice([0,1,2], p=self.probs)
        else:

            task_switch = 0

        mode_data = self.data[self._mode]
        subdomain = np.random.choice(mode_data.keys())
        data = mode_data[subdomain][self._classes_in_task]
        n_classes = len(data)
        self._samples_in_class = data.size(1)
        self._classes_in_task = np.random.choice(np.arange(n_classes), self.n_way,
                    replace=False)


        # sample indices for meta train
        idx = torch.Tensor(self.n_way, self.n_shots)#.to(self.device)
        idx = idx.uniform_(0, self._samples_in_class).long()
        idx = idx.to(self.device)
        data = select_from_tensor(data, idx)

        # build label tensors
        labels = torch.arange(self.n_way).view(-1, 1).expand(-1, self.n_shots).to(self.device)

        # squeeze
        data = data.view(-1, *self.input_size)
        labels = labels.flatten()


        return data, labels, task_switch, self.modes[self._mode]
