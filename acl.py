# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import sys, time, os
import numpy as np
import torch
import copy
import utils
import pdb
from copy import deepcopy
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, r2_score
sys.path.append('../')

from networks.discriminator import Discriminator

class ACL(object):

    def __init__(self, model, args, network):
        self.args=args

        self.sbatch=args.batch_size

#         # optimizer & adaptive lr



        self.e_lr=[args.e_lr] * args.num_domains
        self.d_lr=[args.d_lr] * args.num_domains


        self.lr_min=args.lr_min
        self.lr_factor=args.lr_factor
        self.lr_patience=args.lr_patience



        self.device=args.device
        self.checkpoint=args.checkpoint

        self.adv_loss_reg=args.adv
        self.diff_loss_reg=args.orth

#         # online parameter
        self.last_train_acc = 0
        self.s_update = True
        self.update_metric = args.update_metric
        self.last_train_loss = np.nan
        self.total_batch_count = 0
        self.loss_threshold_interval = args.loss_threshold_interval
        self.acc_threshold_interval = args.acc_threshold_interval

#         # network structure
        self.diff=args.diff

        self.network=network
        self.input_size=args.input_size
        




        # Initialize generator and discriminator
        self.model=model
        self.discriminator=self.get_discriminator(0)
        self.discriminator.get_size()

        self.hidden_size=args.hidden_size
        self.is_classification_task = args.is_classification_task

        if args.is_classification_task:
            self.task_loss=torch.nn.CrossEntropyLoss().to(self.device)
            
            
        else:
            self.task_loss = torch.nn.MSELoss().to(self.device)

            
        self.adversarial_loss_s=torch.nn.CrossEntropyLoss().to(self.device)
        self.adversarial_loss_d=torch.nn.CrossEntropyLoss().to(self.device)
        self.diff_loss=DiffLoss().to(self.device)

        self.optimizer_S=self.get_S_optimizer(0)
        self.optimizer_D=self.get_D_optimizer(0)

        self.task_encoded={}

        self.mu=0.0
        self.sigma=1.0

#         #logging
        self.wandb_log = args.wandb
        if self.wandb_log:
            import wandb

            self.wandb = wandb
        

        print()

    def get_discriminator(self, task_id):
        discriminator=Discriminator(self.args, task_id).to(self.args.device)
        return discriminator

    def get_S_optimizer(self, task_id, e_lr=None):
        if e_lr is None: e_lr=self.e_lr[task_id]
        optimizer_S=torch.optim.SGD(self.model.parameters(), momentum=self.args.mom,
                                    weight_decay=self.args.e_wd, lr=e_lr)
        return optimizer_S

    def get_D_optimizer(self, task_id, d_lr=None):
        if d_lr is None: d_lr=self.d_lr[task_id]
        optimizer_D=torch.optim.SGD(self.discriminator.parameters(), weight_decay=self.args.d_wd, lr=d_lr)
        return optimizer_D



    def train(self, task_id, dataset):

        self.discriminator=self.get_discriminator(self.args.num_domains_pretrain )



        d_lr=self.d_lr[task_id]
        self.optimizer_D=self.get_D_optimizer(task_id, d_lr)
        e_lr=self.e_lr[task_id]       
        self.optimizer_S=self.get_S_optimizer(task_id, e_lr)
        
        
        # parameters for lr adaptation and early stopping
        best_loss=np.inf
        best_model=utils.get_model(self.model)

        best_loss_d=np.inf
        best_model_d=utils.get_model(self.discriminator)

        
        patience=self.lr_patience
        patience_d=self.lr_patience
        dis_lr_update=True


        for i in range(self.args.num_epochs):
            #print (i)

            # Train      
            self.train_epoch(dataset['train'], task_id)
            self.train_res=self.eval_(dataset['train'], task_id)
        
            # Valid
            self.valid_res=self.eval_(dataset['valid'], task_id)

            # Log
            print (self.train_res['loss_tot'],self.train_res['acc_measure'],self.valid_res['loss_tot'],self.valid_res['acc_measure'])
            if self.wandb_log:
                utils.log_wandb(self.train_res,i,'pre-train_train')
                utils.log_wandb(self.valid_res,i,'pre-train_valid')
           

            # Adapt lr for S and D
            if self.valid_res['loss_tot'] < best_loss:
                best_loss=self.valid_res['loss_tot']
                best_model=utils.get_model(self.model)
                patience=self.lr_patience
                #print(' *', end='')
            else:
                patience-=1
                if patience <= 0:
                    e_lr/=self.lr_factor
                    print(' lr={:.1e}'.format(e_lr), end='')
                    if e_lr < self.lr_min:
                        print()
                        break
                    patience=self.lr_patience
                    self.optimizer_S=self.get_S_optimizer(task_id, e_lr)
    
            if self.train_res['loss_a'] < best_loss_d:
                best_loss_d=self.train_res['loss_a']
                best_model_d=utils.get_model(self.discriminator)
                patience_d=self.lr_patience
            else:
                patience_d-=1
                if patience_d <= 0 and dis_lr_update:
                    d_lr/=self.lr_factor
                    print(' Dis lr={:.1e}'.format(d_lr))
                    if d_lr < self.lr_min:
                        dis_lr_update=False
                        print("Dis lr reached minimum value")
                        print()
                        break
                    patience_d=self.lr_patience
                    self.optimizer_D=self.get_D_optimizer(task_id, d_lr)
            #print()
    
        # Restore best validation model (early-stopping)
        self.model.load_state_dict(copy.deepcopy(best_model))
        self.discriminator.load_state_dict(copy.deepcopy(best_model_d))

        self.save_all_models(task_id)

    def train_cl(self, cl_loader):
        self.discriminator=self.get_discriminator(3)
        
        for count, (data, target, tt, td,mode) in enumerate(cl_loader):
            data = data[0]
            target = target[0]
            tt = tt[0]
            td = td[0]
            mode = mode[0]


            
            self.model.train()
            self.discriminator.train()
            t_current=mode
            task_id = mode[0]
            body_mask=torch.eq(t_current, tt).cpu().numpy()
            x=data.to(device=self.device)
            if self.is_classification_task:
                y=target.to(device=self.device, dtype=torch.long)
            else:
                y=target.to(device=self.device, dtype=torch.float32)
            tt=tt.to(device=self.device)

            # Detaching samples in the batch which do not belong to the current task before feeding them to P
            t_current=task_id * torch.ones_like(tt)
            body_mask=torch.eq(t_current, tt).cpu().numpy()
            # x_task_module=data.to(device=self.device)
            x_task_module=data.clone()
            for index in range(x.size(0)):
                
                if body_mask[index] == 0:
                    print (index)
                    x_task_module[index]=x_task_module[index].detach()
            x_task_module=x_task_module.to(device=self.device)

            # Discriminator's real and fake task labels
            t_real_D=td.to(self.device, dtype=torch.long)
            t_fake_D=torch.zeros_like(t_real_D).to(self.device, dtype=torch.long)


            # ================================================================== #
            #                        Train Shared Module                          #
            # ================================================================== #
            # training S for s_steps
            




            self.optimizer_S.zero_grad()
            self.model.zero_grad()
            self.model.freeze_s_module(freeze=not self.s_update)
            
            #pdb.set_trace()
            output=self.model(x, x_task_module, tt, task_id)


            
            task_loss=self.task_loss(output, y)

            shared_encoded, task_encoded=self.model.get_encoded_ftrs(x, x_task_module, task_id)
            dis_out_gen_training=self.discriminator.forward(shared_encoded, t_real_D, task_id)

             

            adv_loss=self.adversarial_loss_s(dis_out_gen_training, t_real_D)

            
            if self.diff == 'yes':
                diff_loss=self.diff_loss(shared_encoded, task_encoded)
            else:
                diff_loss=torch.tensor(0).to(device=self.device, dtype=torch.float32)
                self.diff_loss_reg=0

            total_loss=task_loss + self.adv_loss_reg * adv_loss + self.diff_loss_reg * diff_loss

            #pdb.set_trace() 
            print (total_loss)
            total_loss.backward(retain_graph=True)
            
            
            self.optimizer_S.step()
            print (total_loss)

            # ================================================================== #
            #                          Train Discriminator                       #
            # ================================================================== #
            # training discriminator for d_steps


            self.optimizer_D.zero_grad()
            self.discriminator.zero_grad()

            # training discriminator on real data
            output=self.model(x, x_task_module, tt, task_id)
            shared_encoded, task_out=self.model.get_encoded_ftrs(x, x_task_module, task_id)
            dis_real_out=self.discriminator.forward(shared_encoded.detach(), t_real_D, task_id)
            dis_real_loss=self.adversarial_loss_d(dis_real_out, t_real_D)
            if self.args.dataset == 'miniimagenet':
                dis_real_loss*=self.adv_loss_reg
            dis_real_loss.backward(retain_graph=True)

            # training discriminator on fake data
            z_fake=torch.as_tensor(np.random.normal(self.mu, self.sigma, (x.size(0), self.hidden_size)),dtype=torch.float32, device=self.device)
            dis_fake_out=self.discriminator.forward(z_fake, t_real_D, task_id)
            dis_fake_loss=self.adversarial_loss_d(dis_fake_out, t_fake_D)
            if self.args.dataset == 'miniimagenet':
                dis_fake_loss*=self.adv_loss_reg
            dis_fake_loss.backward(retain_graph=True)

            self.optimizer_D.step()
            
            self.model.eval()
            self.discriminator.eval()
            output=self.model(x, x, tt, task_id)
            r2 = r2_score(y.detach().numpy(),output.detach().numpy())
            print (f'r2 score for task {count} is {r2}')

            
            


            self.total_batch_count +=1
            if count==self.args.timesteps-1:
                break
            
    def train_epoch(self, train_loader, task_id):

        self.model.train()
        self.discriminator.train()

        for count, (data, target, tt, td) in enumerate(train_loader):
            

            task_id = np.argmax(np.bincount(tt.numpy()))
            if self.is_classification_task:
                data = data.view(data.shape[0]*data.shape[1],data.shape[2],data.shape[3],data.shape[4])
                target = target.view(-1)
                y = target.to(device=self.device, dtype=torch.long)
                tt = tt.view(-1).to(device=self.device, dtype=torch.int)
                td = td.view(-1).to(device=self.device, dtype=torch.long)
                #print (data.shape, target, tt,td)
            else:
                y = target.to(device=self.device, dtype=torch.float32)
                tt = tt.to(device=self.device)
                
                
            # data to device
            x = data.to(device=self.device)
            task_id = np.argmax(np.bincount(tt.cpu().numpy()))

            
            
            # Detaching samples in the batch which do not belong to the current task before feeding them to P
            t_current = task_id * torch.ones_like(tt)

            body_mask = torch.eq(t_current, tt).cpu().numpy()
            x_task_module = data.clone()
        
            for index in range(x.size(0)):
                if body_mask[index] == 0:
                    x_task_module[index]=x_task_module[index].detach()
            x_task_module=x_task_module.to(device=self.device)


            # Discriminator's real and fake task labels
            t_real_D=td.to(self.device)
            t_fake_D=torch.zeros_like(t_real_D).to(self.device, dtype=torch.long)


            # ================================================================== #
            #                        Train Shared Module                          #
            # ================================================================== #
            # training S for s_steps
            self.optimizer_S.zero_grad()
            self.model.zero_grad()
            
            #print (x.shape)
            output=self.model(x, x_task_module, tt, task_id) 
            shared_encoded, task_encoded=self.model.get_encoded_ftrs(x, x_task_module, task_id)
            dis_out_gen_training=self.discriminator.forward(shared_encoded, t_real_D, task_id)
            
            #print (output,y)
            task_loss=self.task_loss(output, y)            
            adv_loss=self.adversarial_loss_s(dis_out_gen_training, t_real_D)

            
            if self.diff == 'yes':
                diff_loss=self.diff_loss(shared_encoded, task_encoded)
            else:
                diff_loss=torch.tensor(0).to(device=self.device, dtype=torch.float32)
                self.diff_loss_reg=0

            total_loss=task_loss + self.adv_loss_reg * adv_loss + self.diff_loss_reg * diff_loss



            # print (task_loss.data)#,self.adv_loss_reg * adv_loss,self.diff_loss_reg * diff_loss)
            total_loss.backward(retain_graph=True)
            # for n,para in self.model.named_parameters():
            #     print (f'{n}, weight {para}, grad{para.grad} ')
                
                
            # pdb.set_trace() 
            self.optimizer_S.step()
            output=self.model(x, x_task_module, tt, task_id) 
            task_loss=self.task_loss(output, y) 
            # print (task_loss.data)
            # print()
            





            # ================================================================== #
            #                          Train Discriminator                       #
            # ================================================================== #
            # training discriminator for d_steps


            self.optimizer_D.zero_grad()
            self.discriminator.zero_grad()

            # training discriminator on real data
            output=self.model(x, x_task_module, tt, task_id)
            shared_encoded, task_out=self.model.get_encoded_ftrs(x, x_task_module, task_id)
            dis_real_out=self.discriminator.forward(shared_encoded.detach(), t_real_D, task_id)
            dis_real_loss=self.adversarial_loss_d(dis_real_out, t_real_D)
            if self.args.dataset == 'miniimagenet':
                dis_real_loss*=self.adv_loss_reg
            dis_real_loss.backward(retain_graph=True)

            # training discriminator on fake data
            z_fake=torch.as_tensor(np.random.normal(self.mu, self.sigma, (x.size(0), self.hidden_size)),dtype=torch.float32, device=self.device)
            dis_fake_out=self.discriminator.forward(z_fake, t_real_D, task_id)
            dis_fake_loss=self.adversarial_loss_d(dis_fake_out, t_fake_D)
            if self.args.dataset == 'miniimagenet':
                dis_fake_loss*=self.adv_loss_reg
            dis_fake_loss.backward(retain_graph=True)

            self.optimizer_D.step()


            self.total_batch_count +=1

        


        return

    def if_s_update(self,last_acc, cur_acc, last_loss, cur_loss, task_id):
        if task_id == 0:
            self.s_update = True

            print ('upadte shared module')
            return
        elif self.update_metric == 'loss':
            if cur_loss > self.loss_threshold or cur_loss > last_loss * self.loss_threshold_interval:
                self.s_update = True 
                print ('upadte shared module')
                return
        elif self.update_metric == 'acc':
            if cur_acc < self.acc_threshold or cur_acc < last_acc - self.acc_threshold_interval:
                self.s_update = True 
                print ('upadte shared module')
                return
        self.s_update = False 
        return 


    def eval_batch_classification(self,data,target,tt,td,task_id,model=None):
        if not model:
            model = self.model
        x=data.to(device=self.device)
        y=target.to(device=self.device, dtype=torch.long)
        tt=tt.to(device=self.device)
        t_real_D=td.to(self.device)

        # Forward
        output=model(x, x, tt, task_id)
        
        #print (x.view(x.size(0), -1))
        #pdb.set_trace()
        shared_out, task_out=model.get_encoded_ftrs(x, x, task_id)
        _, pred=output.max(1)
        # import matplotlib.pyplot as plt
        # for img in x[:,0]:
        #     img = img.numpy() 
        #     plt.imshow(img)
        #     plt.show()
            
        correct_t=pred.eq(y.view_as(pred)).sum().item()

        # Discriminator's performance:
        output_d=self.discriminator.forward(shared_out, t_real_D, task_id)
        _, pred_d=output_d.max(1)
        correct_d=pred_d.eq(t_real_D.view_as(pred_d)).sum().item()

        # Loss values
        task_loss=self.task_loss(output, y)
        adv_loss=self.adversarial_loss_d(output_d, t_real_D)

        if self.diff == 'yes':
            diff_loss=self.diff_loss(shared_out, task_out)
        else:
            diff_loss=torch.tensor(0).to(device=self.device, dtype=torch.float32)
            self.diff_loss_reg=0

        total_loss = task_loss + self.adv_loss_reg * adv_loss + self.diff_loss_reg * diff_loss
        num = x.size(0)

        return task_loss, adv_loss, diff_loss, total_loss, correct_t, correct_d, num
    
    def eval_batch_regression(self,data,target,tt,td,task_id,model=None):
        if not model:
            model = self.model
        x=data.to(device=self.device)
        y=target.to(device=self.device, dtype=torch.long)
        tt=tt.to(device=self.device)
        t_real_D=td.to(self.device)
        


        # Forward
        output=model(x, x, tt, task_id)
        shared_out, task_out=model.get_encoded_ftrs(x, x, task_id)



        # Discriminator's performance:
        output_d=self.discriminator.forward(shared_out, t_real_D, task_id)
        _, pred_d=output_d.max(1)
        correct_d=pred_d.eq(t_real_D.view_as(pred_d)).sum().item()

        # Loss values
        task_loss=self.task_loss(output, y)
        adv_loss=self.adversarial_loss_d(output_d, t_real_D)

        if self.diff == 'yes':
            diff_loss=self.diff_loss(shared_out, task_out)
        else:
            diff_loss=torch.tensor(0).to(device=self.device, dtype=torch.float32)
            self.diff_loss_reg=0

        total_loss = task_loss + self.adv_loss_reg * adv_loss + self.diff_loss_reg * diff_loss
        num = x.size(0)


        return task_loss, adv_loss, diff_loss, total_loss, output, correct_d, y, num






    def eval_(self, data_loader, task_id):
        loss_a, loss_t, loss_d, loss_total=0, 0, 0, 0
        correct_d, correct_t = 0, 0
        num=0
        batch=0
        r2_output = []
        r2_y = []

        self.model.eval()
        self.discriminator.eval()

        res={}

        with torch.no_grad():
            for batch, (data, target, tt, td) in enumerate(data_loader):
                if self.is_classification_task:
                    data = data.view(data.shape[0]*data.shape[1],data.shape[2],data.shape[3],data.shape[4])
                    target = target.view(-1)
                    y = target.to(device=self.device, dtype=torch.long)
                    tt = tt.view(-1).to(device=self.device, dtype=torch.int)
                    td = td.view(-1).to(device=self.device, dtype=torch.long)
                    #print (data.shape, target, tt,td)

                    task_loss, adv_loss, diff_loss, total_loss, correct_t_batch, correct_d_batch, num_batch = self.eval_batch_classification(data,target,tt,td,task_id)
                    correct_t += correct_t_batch

                    
                else:
                    y = target.to(device=self.device, dtype=torch.float32)
                    tt = tt.to(device=self.device)

                    task_loss, adv_loss, diff_loss, total_loss, output, correct_d_batch, y, num_batch = self.eval_batch_regression(data,target,tt,td,task_id)
                    r2_output.extend(output)
                    r2_y.extend(y)
                    
                loss_t += task_loss
                loss_a += adv_loss
                loss_d += diff_loss
                loss_total += total_loss
                correct_d += correct_d_batch
                num += num_batch
                

        
        if not self.is_classification_task:
            res['loss_t'], res['acc_measure']=loss_t.item() / (batch + 1), r2_score(r2_y,r2_output)
        else:
            res['loss_t'], res['acc_measure']=loss_t.item() / (batch + 1), 100 * correct_t / num
        res['loss_a'], res['acc_d']=loss_a.item() / (batch + 1), 100 * correct_d / num
        res['loss_d']=loss_d.item() / (batch + 1)
        res['loss_tot']=loss_total.item() / (batch + 1)
        res['size']=self.loader_size(data_loader)


        return res


    def test(self, data_loader, task_id, model):
        loss_a, loss_t, loss_d, loss_total=0, 0, 0, 0
        correct_d, correct_t=0, 0
        num=0
        batch=0

        model.eval()
        self.discriminator.eval()

        res={}
        r2o = []
        r2y = []
        with torch.no_grad():
            for batch, (data, target, tt, td) in enumerate(data_loader):
                if self.is_classification_task:
                    data = data.view(data.shape[0]*data.shape[1],data.shape[2],data.shape[3],data.shape[4])
                    target = target.view(-1)
                    y = target.to(device=self.device, dtype=torch.long)
                    tt = tt.view(-1).to(device=self.device, dtype=torch.int)
                    td = td.view(-1).to(device=self.device, dtype=torch.long)
                    #print (data.shape, target, tt,td)

                    task_loss, adv_loss, diff_loss, total_loss, correct_t_batch, correct_d_batch, num_batch = self.eval_batch_classification(data,target,tt,td,task_id,model=model)
                    correct_t += correct_t_batch

                    
                else:
                    y = target.to(device=self.device, dtype=torch.float32)
                    tt = tt.to(device=self.device)

                    task_loss, adv_loss, diff_loss, total_loss, output, correct_d_batch, y, num_batch = self.eval_batch_regression(data,target,tt,td,task_id,model=model)
                    r2_output.extend(output)
                    r2_y.extend(y)

                loss_t += task_loss
                loss_a += adv_loss
                loss_d += diff_loss
                loss_total += total_loss
                correct_d += correct_d_batch
                num += num_batch


        if not self.is_classification_task:
            res['loss_t'], res['acc_measure']=loss_t.item() / (batch + 1), r2_score(r2_y,r2_output)
        else:
            res['loss_t'], res['acc_measure']=loss_t.item() / (batch + 1), 100 * correct_t / num
        res['loss_a'], res['acc_d']=loss_a.item() / (batch + 1), 100 * correct_d / num
        res['loss_d']=loss_d.item() / (batch + 1)
        res['loss_tot']=loss_total.item() / (batch + 1)
        res['size']=self.loader_size(data_loader)

        return res



    def save_all_models(self, task_id):
        print("Saving all models for task {} ...".format(task_id+1))
        dis=utils.get_model(self.discriminator)
        torch.save({'model_state_dict': dis,
                    }, os.path.join(self.checkpoint, 'discriminator_{}.pth.tar'.format(task_id)))

        model=utils.get_model(self.model)
        torch.save({'model_state_dict': model,
                    }, os.path.join(self.checkpoint, 'model_{}.pth.tar'.format(task_id)))



    def load_model(self, task_id):

        # Load a previous model
        net=self.network.Net(self.args)
        checkpoint=torch.load(os.path.join(self.checkpoint, 'model_{}.pth.tar'.format(task_id)))
        net.load_state_dict(checkpoint['model_state_dict'])

        # # Change the previous shared module with the current one
        current_shared_module=deepcopy(self.model.shared.state_dict())
        net.shared.load_state_dict(current_shared_module)

        net=net.to(self.args.device)
        return net


    def load_checkpoint(self, task_id):
        print("Loading checkpoint for task {} ...".format(task_id))

        # Load a previous model
        net=self.network.Net(self.args)
        checkpoint=torch.load(os.path.join(self.checkpoint, 'model_{}.pth.tar'.format(task_id)))
        net.load_state_dict(checkpoint['model_state_dict'])
        net=net.to(self.args.device)
        return net


    def loader_size(self, data_loader):
        return data_loader.dataset.__len__()



    def get_tsne_embeddings_first_ten_tasks(self, dataset, model):
        from tensorboardX import SummaryWriter

        model.eval()

        tag_ = '_diff_{}'.format(self.args.diff)
        all_images, all_shared, all_private = [], [], []

        # Test final model on first 10 tasks:
        writer = SummaryWriter()
        for t in range(10):
            for itr, (data, _, tt, td) in enumerate(dataset[t]['tsne']):
                x = data.to(device=self.device)
                tt = tt.to(device=self.device)
                output = model.forward(x, x, tt, t)
                shared_out, private_out = model.get_encoded_ftrs(x, x, t)
                all_shared.append(shared_out)
                all_private.append(private_out)
                all_images.append(x)

        print (torch.stack(all_shared).size())


        tag = ['Shared10_{}_{}'.format(tag_,i) for i in range(1,11)]
        writer.add_embedding(mat=torch.stack(all_shared,dim=1).data, label_img=torch.stack(all_images,dim=1).data, metadata=list(range(1,11)),
                             tag=tag)#, metadata_header=list(range(1,11)))

        tag = ['Private10_{}_{}'.format(tag_, i) for i in range(1, 11)]
        writer.add_embedding(mat=torch.stack(all_private,dim=1).data, label_img=torch.stack(all_images,dim=1).data, metadata=list(range(1,11)),
                         tag=tag)#,metadata_header=list(range(1,11)))
        writer.close()


    def get_tsne_embeddings_last_three_tasks(self, dataset, model):
        from tensorboardX import SummaryWriter

        # Test final model on last 3 tasks:
        model.eval()
        tag = '_diff_{}'.format(self.args.diff)

        for t in [17,18,19]:
            all_images, all_labels, all_shared, all_private = [], [], [], []
            writer = SummaryWriter()
            for itr, (data, target, tt, td) in enumerate(dataset[t]['tsne']):
                x = data.to(device=self.device)
                y = target.to(device=self.device, dtype=torch.long)
                tt = tt.to(device=self.device)
                output = model.forward(x, x, tt, t)
                shared_out, private_out = model.get_encoded_ftrs(x, x, t)
                # print (shared_out.size())

                all_shared.append(shared_out)
                all_private.append(private_out)
                all_images.append(x)
                all_labels.append(y)

            writer.add_embedding(mat=torch.stack(all_shared,dim=1).data, label_img=torch.stack(all_images,dim=1).data,
                                 metadata=list(range(1,6)), tag='Shared_{}_{}'.format(t, tag))
                                 # ,metadata_header=list(range(1,6)))
            writer.add_embedding(mat=torch.stack(all_private,dim=1).data, label_img=torch.stack(all_images,dim=1).data,
                                 metadata=list(range(1,6)), tag='Private_{}_{}'.format(t, tag))
                                 # ,metadata_header=list(range(1,6)))

        writer.close()



#         #
class DiffLoss(torch.nn.Module):
    # From: Domain Separation Networks (https://arxiv.org/abs/1608.06019)
    # Konstantinos Bousmalis, George Trigeorgis, Nathan Silberman, Dilip Krishnan, Dumitru Erhan

    def __init__(self):
        super(DiffLoss, self).__init__()

    def forward(self, D1, D2):
        D1=D1.view(D1.size(0), -1)
        D1_norm=torch.norm(D1, p=2, dim=1, keepdim=True).detach()
        D1_norm=D1.div(D1_norm.expand_as(D1) + 1e-6)

        D2=D2.view(D2.size(0), -1)
        D2_norm=torch.norm(D2, p=2, dim=1, keepdim=True).detach()
        D2_norm=D2.div(D2_norm.expand_as(D2) + 1e-6)

        # return torch.mean((D1_norm.mm(D2_norm.t()).pow(2)))
        return torch.mean((D1_norm.mm(D2_norm.t()).pow(2)))
