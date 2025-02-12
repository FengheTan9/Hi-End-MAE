import os
import time
import math
import shutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataloader.med_datasets import LargeMedicalDataSets
from hi_end_mae import build_vit_large_hi_end_mae_p12_3d, build_vit_base_hi_end_mae_p16_3d, build_vit_large_hi_end_mae_p16_3d
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



class HiEndMAETrainer(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model_name = 'HiEndMAE'
        dist.init_process_group('nccl', init_method='env://')
        rank = dist.get_rank()   
        local_rank = os.environ['LOCAL_RANK']
        master_addr = os.environ['MASTER_ADDR']  
        master_port = os.environ['MASTER_PORT']  
        print(f"rank = {rank} is initialized in {master_addr}:{master_port}; local_rank = {local_rank}")
        torch.cuda.set_device(rank)
        args.local_rank = rank

        self.scaler = torch.cuda.amp.GradScaler()
        self.init_lr()
        self.build_model()
        self.build_optimizer()
        self.build_dataloader()
        self.iters_per_epoch = len(self.dataloader)
        if not os.path.exists(args.ckpt_dir):
            os.mkdir(args.ckpt_dir)
        print("mask ratio: {}".format(args.mask_ratio))
        if dist.get_rank() == 0:
            print(f"Local Batch Size on GPU {args.local_rank}: {args.batch_size}")
    def init_lr(self):
        args = self.args
        # infer learning rate before changing batch size
        self.lr = args.base_lr

    def build_model(self):
        args = self.args

        if args.model == "hi_end_mae_vit_large_12p":
            model = build_vit_large_hi_end_mae_p12_3d(args=args).to(args.local_rank)
        elif args.model == "hi_end_mae_vit_base_16p":
            model = build_vit_base_hi_end_mae_p16_3d(args=args).to(args.local_rank) 
        elif args.model == "hi_end_mae_vit_large_16p":
            model = build_vit_large_hi_end_mae_p16_3d(args=args).to(args.local_rank) 
        else:
            model = None
        self.model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)
        
    def build_optimizer(self):
        args = self.args

        optim_params = self.get_param_groups(nowd_keys={'cls_token', 'pos_embed', 'mask_token', 'decoder_pos_embed', 'gamma'})
        # TODO: create optimizer factory
        self.optimizer = torch.optim.AdamW(optim_params,
                                            lr=self.lr,
                                            betas=(args.beta1, args.beta2),
                                            weight_decay=args.weight_decay)
    def build_dataloader(self):
        args = self.args
        train_dataset = LargeMedicalDataSets(args=args)
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        self.dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size, num_workers=16, pin_memory=True)

    def run(self):
        args = self.args
        # Compute iterations when resuming
        niters = 0
        for epoch in range(args.remain, args.epochs):
            # train for one epoch
            self.dataloader.sampler.set_epoch(epoch)
            niters = self.epoch_train(epoch, niters)
            if (epoch + 1) % 200 == 0 and dist.get_rank() == 0:
                self.save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scaler': self.scaler.state_dict(),  # additional line compared with base imple
                }, is_best=False, filename=f'{args.save_ckpt_dir}/{args.model}_ckpt_{epoch:04d}.pth')
                print("=> finish saving checkpoint")

    def epoch_train(self, epoch, niters):
        args = self.args
        train_loader = self.dataloader
        model = self.model
        optimizer = self.optimizer
        scaler = self.scaler
        total_loss = AverageMeter()
        load_time = 0
        forward_time = 0
        bp_time = 0
        cache2gpu_time = 0
        # switch to train mode
        model.train()
        load_start_time = time.time()
        start_time = time.time()

        for i, batch_data in enumerate(train_loader):
            load_time += time.time() - load_start_time
            # adjust learning at the beginning of each iteration
            self.adjust_learning_rate(epoch + i / self.iters_per_epoch, args)

            # For SSL pretraining, only image data is required for training
            gpu_time = time.time()
            batch_data = [{k: v.to(args.local_rank) for k, v in crop.items()} for crop in batch_data]
            image = torch.stack([crop["image"] for crop in batch_data], dim=0).view(-1, 1, 96, 96, 96).to(args.local_rank)
            cache2gpu_time += time.time() - gpu_time

            forward_start_time = time.time()
            with torch.cuda.amp.autocast(True):
                loss = model(image, return_image=False)
                total_loss.update(loss.item(), image.size(0))
            forward_time += time.time() - forward_start_time

            # compute gradient and do SGD step
            bp_start_time = time.time()
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            bp_time += time.time() - bp_start_time
            niters += 1
            load_start_time = time.time()

        end_time = time.time()
        duration = end_time - start_time
        # Log to the screen
        if dist.get_rank() == 0:
            print("Epoch: {}/{} took {:.2f} seconds | TotalIter {}/{} | Init Lr: {:.6f} | Lr: {:.6f} | Load Time: {:.2f} s | GPU Time: {:.2f} s | Forward Time: {:.2f} s | Backward Time: {:.2f} s | Loss: {:.4f}"
                  .format(epoch, args.epochs, duration, niters, args.epochs * self.iters_per_epoch, self.lr, optimizer.param_groups[0]['lr'], 
                          load_time, 
                          cache2gpu_time,
                          forward_time, 
                          bp_time, 
                          total_loss.avg))
        return niters

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth'):
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, 'model_best.pth')

    def adjust_learning_rate(self, epoch, args):
        """Base schedule: CosineDecay with warm-up."""
        init_lr = self.lr
        if epoch < args.warmup_epochs:
            cur_lr = init_lr * epoch / args.warmup_epochs
        else:
            cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
        for param_group in self.optimizer.param_groups:
            if 'fix_lr' in param_group and param_group['fix_lr']:
                param_group['lr'] = init_lr
            else:
                param_group['lr'] = cur_lr

    def get_param_groups(self, nowd_keys=()):
        para_groups, para_groups_dbg = {}, {}
    
        for name, para in self.model.named_parameters():
            if not para.requires_grad:
                continue  # frozen weights
            if len(para.shape) == 1 or name.endswith('.bias') or any(k in name for k in nowd_keys):
                wd_scale, group_name = 0., 'no_decay'
            else:
                wd_scale, group_name = 1., 'decay'
            
            if group_name not in para_groups:
                para_groups[group_name] = {'params': [], 'weight_decay_scale': wd_scale, 'lr_scale': 1.}
                para_groups_dbg[group_name] = {'params': [], 'weight_decay_scale': wd_scale, 'lr_scale': 1.}
            para_groups[group_name]['params'].append(para)
            para_groups_dbg[group_name]['params'].append(name)
        
        return list(para_groups.values())
