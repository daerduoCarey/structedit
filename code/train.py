"""
    Conditional VAE for Shape Diff
    Training time: take obj1 + a topk retrieved obj2 to get obj2-obj1 shape_diff
"""

import os
import time
import sys
import shutil
import random
from time import strftime
from argparse import ArgumentParser
import numpy as np
import torch
import torch.utils.data
from config import add_train_vae_args
from data import PartNetShapeDiffDataset, Tree
import utils

# Use 1-4 CPU threads to train.
# Don't use too many CPU threads, which will slow down the training.
torch.set_num_threads(2)

def train(conf):
    # load network model
    models = utils.get_model_module(conf.model_version)

    # check if training run already exists. If so, delete it.
    if conf.resume_epoch is None:
        if os.path.exists(os.path.join(conf.log_path, conf.exp_name)) or \
        os.path.exists(os.path.join(conf.ckpt_path, conf.exp_name)):
            response = input('A training run named "%s" already exists, overwrite? (y/n) ' % (conf.exp_name))
            if response != 'y':
                sys.exit()
        if os.path.exists(os.path.join(conf.log_path, conf.exp_name)):
            shutil.rmtree(os.path.join(conf.log_path, conf.exp_name))
        if os.path.exists(os.path.join(conf.ckpt_path, conf.exp_name)):
            shutil.rmtree(os.path.join(conf.ckpt_path, conf.exp_name))

    # create directories for this run
    if not os.path.exists(os.path.join(conf.ckpt_path, conf.exp_name)):
        os.makedirs(os.path.join(conf.ckpt_path, conf.exp_name))
    if not os.path.exists(os.path.join(conf.log_path, conf.exp_name)):
        os.makedirs(os.path.join(conf.log_path, conf.exp_name))

    # file log
    flog = open(os.path.join(conf.log_path, conf.exp_name, 'train.log'), 'w')

    # backup python files used for this training
    os.system('cp config.py data.py %s.py %s %s' % (conf.model_version, __file__, os.path.join(conf.log_path, conf.exp_name)))

    # set training device
    device = torch.device(conf.device)
    print(f'Using device: {conf.device}')
    flog.write(f'Using device: {conf.device}\n')

    # log the object category information
    print(f'Object Category: {conf.category}')
    flog.write(f'Object Category: {conf.category}\n')

    # control randomness
    if conf.seed < 0:
        conf.seed = random.randint(1, 10000)
    print("Random Seed: %d" % (conf.seed))
    flog.write(f'Random Seed: {conf.seed}\n')
    random.seed(conf.seed)
    np.random.seed(conf.seed)
    torch.manual_seed(conf.seed)

    # save config
    torch.save(conf, os.path.join(conf.ckpt_path, conf.exp_name, 'conf.pth'))

    # create models
    encoder = models.RecursiveEncoder(conf, variational=True, probabilistic=not conf.non_variational)
    decoder = models.RecursiveDecoder(conf)
    models = [encoder, decoder]
    model_names = ['encoder', 'decoder']

    # create optimizers
    encoder_opt = torch.optim.Adam(encoder.parameters(), lr=conf.lr)
    decoder_opt = torch.optim.Adam(decoder.parameters(), lr=conf.lr)
    optimizers = [encoder_opt, decoder_opt]
    optimizer_names = ['encoder', 'decoder']

    if conf.resume_epoch is not None:
        # load pretrained model
        start_epoch = utils.load_checkpoint(
            models=models, model_names=model_names,
            dirname=os.path.join(conf.ckpt_path, conf.exp_name),
            epoch=conf.resume_epoch,
            optimizers=optimizers, optimizer_names=optimizer_names,
            strict=True,
            device=device)
    else:
        start_epoch = 0

    # create training and validation datasets and data loaders
    data_features = ['object', 'diff']
    train_dataset = PartNetShapeDiffDataset(
        data_dir=conf.data_path, object_list=conf.train_dataset, data_features=data_features,
        topk=conf.shapediff_topk, metric=conf.shapediff_metric, self_is_neighbor=conf.self_is_neighbor)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=conf.batch_size, shuffle=True, collate_fn=utils.collate_feats, num_workers=conf.num_workers)
    train_num_batch = len(train_dataloader)

    # learning rate schedulers
    encoder_scheduler = torch.optim.lr_scheduler.StepLR(
        encoder_opt, step_size=conf.lr_decay_every, gamma=conf.lr_decay_by, last_epoch=(start_epoch*train_num_batch)-1)
    decoder_scheduler = torch.optim.lr_scheduler.StepLR(
        decoder_opt, step_size=conf.lr_decay_every, gamma=conf.lr_decay_by, last_epoch=(start_epoch*train_num_batch)-1)

    # create logs
    if not conf.no_console_log:
        header = '     Time    Epoch    Iteration    Progress(%)       LR       BoxLoss   StructLoss  DNTypeLoss  DNBoxLoss  KLDivLoss    L1Loss   TotalLoss'
    if not conf.no_tb_log:
        # https://github.com/lanpa/tensorboard-pytorch
        # from tensorboardX import SummaryWriter
        from torch.utils.tensorboard import SummaryWriter
        tb_writer = SummaryWriter(log_dir=os.path.join(conf.log_path, conf.exp_name, 'train'))

    # send parameters to device
    for m in models:
        m.to(device)
    for o in optimizers:
        utils.optimizer_to_device(o, device)

    # start training
    print("Starting training ...... ")
    flog.write('Starting training ......\n')

    start_time = time.time()
    if conf.resume_epoch is None:
        last_checkpoint_step = None
    else:
        last_checkpoint_step = start_epoch * train_num_batch
    last_train_console_log_step = None

    # train for every epoch
    for epoch in range(start_epoch, conf.epochs):
        if not conf.no_console_log:
            print(f'training run {conf.exp_name}')
            flog.write(f'training run {conf.exp_name}\n')
            print(header)
            flog.write(header+'\n')

        train_batches = enumerate(train_dataloader, 0)
        train_fraction_done = 0.0

        # train for every batch
        for train_batch_ind, batch in train_batches:
            train_fraction_done = (train_batch_ind + 1) / train_num_batch
            train_step = epoch * train_num_batch + train_batch_ind

            log_console = not conf.no_console_log and (last_train_console_log_step is None or \
                    train_step - last_train_console_log_step >= conf.console_log_interval)
            if log_console:
                last_train_console_log_step = train_step

            # set models to training mode
            for m in models:
                m.train()

            # forward pass (including logging)
            total_loss = forward(
                batch=batch, data_features=data_features, encoder=encoder, decoder=decoder, device=device, conf=conf,
                step=train_step, epoch=epoch, batch_ind=train_batch_ind, num_batch=train_num_batch, start_time=start_time,
                log_console=log_console, log_tb=not conf.no_tb_log, tb_writer=tb_writer,
                lr=encoder_opt.param_groups[0]['lr'], flog=flog)

            # optimize one step
            encoder_opt.zero_grad()
            decoder_opt.zero_grad()
            total_loss.backward()
            if False:
                for name, param in encoder.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        print(name, torch.norm(param.grad.data))
                for name, param in decoder.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        print(name, torch.norm(param.grad.data))
                exit(1)
            encoder_opt.step()
            decoder_opt.step()
            encoder_scheduler.step()
            decoder_scheduler.step()

            # save checkpoint
            with torch.no_grad():
                if last_checkpoint_step is None or \
                        train_step - last_checkpoint_step >= conf.checkpoint_interval:
                    print("Saving checkpoint ...... ", end='', flush=True)
                    flog.write("Saving checkpoint ...... ")
                    utils.save_checkpoint(
                        models=models, model_names=model_names, dirname=os.path.join(conf.ckpt_path, conf.exp_name),
                        epoch=epoch, prepend_epoch=True, optimizers=optimizers, optimizer_names=model_names)
                    print("DONE")
                    flog.write("DONE\n")
                    last_checkpoint_step = train_step

    # save the final models
    print("Saving final checkpoint ...... ", end='', flush=True)
    flog.write("Saving final checkpoint ...... ")
    utils.save_checkpoint(
        models=models, model_names=model_names, dirname=os.path.join(conf.ckpt_path, conf.exp_name),
        epoch=epoch, prepend_epoch=False, optimizers=optimizers, optimizer_names=optimizer_names)
    print("DONE")
    flog.write("DONE\n")

    flog.close()

def forward(batch, data_features, encoder, decoder, device, conf,
            step=None, epoch=None, batch_ind=0, num_batch=1, start_time=0,
            log_console=False, log_tb=False, tb_writer=None, lr=None, flog=None):

    objects = batch[data_features.index('object')]
    diffs = batch[data_features.index('diff')]

    losses = {
        'box': torch.zeros(1, device=device),
        'leaf': torch.zeros(1, device=device),
        'exists': torch.zeros(1, device=device),
        'semantic': torch.zeros(1, device=device),
        'kldiv': torch.zeros(1, device=device),
        'diffnode_type': torch.zeros(1, device=device),
        'diffnode_box': torch.zeros(1, device=device),
        'l1': torch.zeros(1, device=device),
    }

    # process every data in the batch individually
    for i in range(len(objects)):
        obj = objects[i]
        obj.to(device)
        diff = diffs[i]
        diff.to(device)

        # get part feature for each subtree
        encoder.encode_tree(obj)

        # encode cond_obj and tree_diff
        root_code = encoder.encode_tree_diff(obj, diff)

        # get kldiv loss
        if not conf.non_variational:
            root_code, obj_kldiv_loss = torch.chunk(root_code, 2, 1)
            obj_kldiv_loss = -obj_kldiv_loss.sum() # negative kldiv, sum over feature dimensions
            losses['kldiv'] = losses['kldiv'] + obj_kldiv_loss

        # decode root code to get reconstruction loss
        obj_losses = decoder.tree_diff_recon_loss(root_code, obj, diff)
        for loss_name, loss in obj_losses.items():
            losses[loss_name] = losses[loss_name] + loss

    for loss_name in losses.keys():
        losses[loss_name] = losses[loss_name] / len(objects)

    losses['box'] *= conf.loss_weight_box
    losses['leaf'] *= conf.loss_weight_leaf
    losses['exists'] *= conf.loss_weight_exists
    losses['semantic'] *= conf.loss_weight_semantic
    losses['kldiv'] *= conf.loss_weight_kldiv
    losses['diffnode_type'] *= conf.loss_weight_diffnode_type
    losses['diffnode_box'] *= conf.loss_weight_diffnode_box
    losses['l1'] *= conf.loss_weight_l1

    total_loss = 0
    for loss in losses.values():
        total_loss += loss

    with torch.no_grad():
        # log to console
        if log_console:
            print(
                f'''{strftime("%H:%M:%S", time.gmtime(time.time()-start_time)):>9s} '''
                f'''{epoch:>5.0f}/{conf.epochs:<5.0f} '''
                f'''{batch_ind:>5.0f}/{num_batch:<5.0f} '''
                f'''{100. * (1+batch_ind+num_batch*epoch) / (num_batch*conf.epochs):>9.1f}%      '''
                f'''{lr:>5.2E} '''
                f'''{losses['box'].item():>11.5f} '''
                f'''{(losses['leaf']+losses['exists']+losses['semantic']).item():>11.5f} '''
                f'''{losses['diffnode_type'].item():>10.5f} '''
                f'''{losses['diffnode_box'].item():>10.5f} '''
                f'''{losses['kldiv'].item():>10.5f} '''
                f'''{losses['l1'].item() if 'l1' in losses else 0:>10.5f} '''
                f'''{total_loss.item():>10.5f}''')
            flog.write(
                f'''{strftime("%H:%M:%S", time.gmtime(time.time()-start_time)):>9s} '''
                f'''{epoch:>5.0f}/{conf.epochs:<5.0f} '''
                f'''{batch_ind:>5.0f}/{num_batch:<5.0f} '''
                f'''{100. * (1+batch_ind+num_batch*epoch) / (num_batch*conf.epochs):>9.1f}%      '''
                f'''{lr:>5.2E} '''
                f'''{losses['box'].item():>11.5f} '''
                f'''{(losses['leaf']+losses['exists']+losses['semantic']).item():>11.5f} '''
                f'''{losses['diffnode_type'].item():>10.5f} '''
                f'''{losses['diffnode_box'].item():>10.5f} '''
                f'''{losses['kldiv'].item():>10.5f} '''
                f'''{losses['l1'].item() if 'l1' in losses else 0:>10.5f} '''
                f'''{total_loss.item():>10.5f}\n''')
            flog.flush()

        # log to tensorboard
        if log_tb and tb_writer is not None:
            tb_writer.add_scalar(tag='loss', scalar_value=total_loss.item(), global_step=step)
            tb_writer.add_scalar(tag='lr', scalar_value=lr, global_step=step)
            tb_writer.add_scalar(tag='box_loss', scalar_value=losses['box'].item(), global_step=step)
            tb_writer.add_scalar(tag='leaf_loss', scalar_value=losses['leaf'].item(), global_step=step)
            tb_writer.add_scalar(tag='exists_loss', scalar_value=losses['exists'].item(), global_step=step)
            tb_writer.add_scalar(tag='semantic_loss', scalar_value=losses['semantic'].item(), global_step=step)
            tb_writer.add_scalar(tag='kldiv_loss', scalar_value=losses['kldiv'].item(), global_step=step)
            tb_writer.add_scalar(tag='l1', scalar_value=losses['kldiv'].item(), global_step=step)
            tb_writer.add_scalar(tag='diffnode_type_loss', scalar_value=losses['diffnode_type'].item(), global_step=step)
            tb_writer.add_scalar(tag='diffnode_box_loss', scalar_value=losses['diffnode_box'].item(), global_step=step)

    return total_loss

if __name__ == '__main__':
    sys.setrecursionlimit(5000) # this code uses recursion a lot for code simplicity

    parser = ArgumentParser()
    parser = add_train_vae_args(parser)
    config = parser.parse_args()

    Tree.load_category_info(config.category)
    train(config)

