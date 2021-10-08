import torch, math, time
import numpy as np
import utils


import pdb



# args
from args import parse_args
args = parse_args()


args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

np.random.seed(args.seed)
torch.manual_seed(args.seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    # Faster run but not deterministic:
    # torch.backends.cudnn.benchmark = True
    # To get deterministic results that match with paper at cost of lower speed:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

args.wandb = False
# log
if args.wandb:
    def wandb_wrapper(args, first_time=True):
        # wandb

        if first_time:
            import wandb
            if args.wandb_key is not None:
                wandb.login(key=args.wandb_key)


        wandb.init(project=args.wandb_project, name=args.wandb_name, group=args.wandb_group, reinit=True)
        wandb.config.update(args)

    wandb = wandb_wrapper(args)

else:
    wandb = None





# dataloader
if args.dataset == 'sinusoid':
    from dataloaders.sinusoid import init_dataloaders
elif args.dataset == 'omniglot':  
    from dataloaders.omniglot import init_dataloaders
elif args.dataset == 'tiered_imagenet':
    from dataloaders.tiered_imagenet import init_dataloaders
else:
    raise NotImplementedError('Unknown dataset `{0}`.'.format(args.dataset))

meta_dataloaders, cl_dataloader = init_dataloaders(args)




# agent and network
from acl import ACL as approach
if args.dataset == 'sinusoid' or 'omniglot':
    from networks import mlp_acl as network
elif args.dataset == 'tiered_imagenet':
    print ('from networks import alexnet_acl as network')
    from networks import alexnet_acl as network

net = network.Net(args)
net = net.to(args.device)
net.print_model_size(wandb=wandb)
appr = approach(net,args,network=network)



args.num_epochs = 100

# pretrain
best_val = 0.
epochs_overfitting = 0
epoch_desc = 'Epoch {{0: <{0}d}}'.format(1 + int(math.log10(args.num_epochs)))


    

appr.train(0,meta_dataloaders[0])


meta_test_model = appr.load_model(0)
test_res = appr.test(meta_dataloaders[0]['test'], 0, model=meta_test_model)
print ()
print('>>> Test on task {:2s} - {:15s}: loss={:.3f}, acc_measure={:5.8f} <<<'.format('meta', meta_dataloaders[0]['name'],
                                                                                  test_res['loss_t'],
                                                                                  test_res['acc_measure']))

args.wandb and utils.log_wandb(test_res,0,'pre-train_test')




print('\npretraining done!\n')
#pdb.set_trace()




#appr.train_cl(cl_dataloader)