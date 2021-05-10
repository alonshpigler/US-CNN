from __future__ import print_function
import sys
import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.optim as optim
import warnings
import datetime
import argparse
from model_layer.US_CNN import *
import os

from utils import vis_utils
from utils.timer import Timer

warnings.filterwarnings("ignore")
# from livelossplot import PlotLosses
# import Sandboxes.Alon.US.signal_processing as signal
print('__Python VERSION:', sys.version)
print('__pyTorch VERSION:', torch.__version__)
print('__CUDA VERSION')
print('__CUDNN VERSION:', torch.backends.cudnn.version())
print('__Number CUDA Devices:', torch.cuda.device_count())
print('__Devices')
# call(["nvidia-smi", "--format=csv", "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free"])
print('Active CUDA Device: GPU', torch.cuda.current_device())
print('Available devices ', torch.cuda.device_count())
print('Current cuda device ', torch.cuda.current_device())
use_cuda = torch.cuda.is_available()
print("USE CUDA=" + str(use_cuda))
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
Tensor = FloatTensor


def run_batch(model, data, phase, target, l1_weight=0, criterion='', optimizer=''):
    # data, target = Variable(batch["image"].cuda()), Variable(batch["mask"].cuda())
    if phase == 'train':
        # zero the parameter gradients
        optimizer.zero_grad()

    # forward
    output = model.forward(data)
    ce_loss = criterion(output.float().squeeze(), target)
    loss = ce_loss + l1_weight * torch.norm(output.squeeze(), 1) / output.shape[0]

    # backward
    if phase == 'train':
        loss.backward();
        optimizer.step();

    return output.data.cpu().numpy(), loss.data.item()


def test(dataloader, net_state_dict=[], net_args={}):
    # TODO - support uploading net
    # if len(net_state_dict) > 0:
    net = CNN1D(1, **net_args)
    net.load_state_dict(net_state_dict)
    # else:
    # net = load_network(model_path, **net_args)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    net.to(device)
    net.eval()
    timer = Timer('Executing US-CNN on GPU...').start()
    for batch_idx, batch_data in enumerate(dataloader):
        batch_data = batch_data['input']
        batch_pred = net.forward(batch_data).squeeze()
        batch_pred = np.array(batch_pred.tolist())

        if batch_idx == 0:
            pred = batch_pred
        else:
            pred = np.concatenate((pred, batch_pred))
    timer.end()
    print_runtime = True
    if print_runtime:
        print('US-CNN ran for elapsed time of ' + str(timer.end_time - timer.start_time))
    return pred


def fit(dataloaders,
        net_args,
        res_path='./',
        epochs=30,
        lr=0.01,
        lr__reduce_patience=10,
        early_stopping_patience=25,
        l1_weight=0,
        show_training=True,
        lr_reduce=1,
        save_training=False,
        save_losses=False):

    net = CNN1D(1, **net_args)

    # Initialize weights of first layer to pulses
    # initialize_weights_to_pulses(signal_data, model)

    # set to gpu if any
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    net.to(device)

    optimizer = optim.Adam(net.parameters(), lr)
    if lr_reduce:
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=lr__reduce_patience, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=0,
                                                                  min_lr=0.00001,
                                                                  eps=1e-08)
    criterion = nn.MSELoss()

    train_losses = []
    val_losses = []
    best_val_loss = math.inf
    bad_epochs = 0
    stop = 0
    # early_stopping = EarlyStopping(early_stopping_patience)

    for epoch in range(1, epochs + 1):
        for phase in ['train', 'val']:
            epoch_loss = 0
            if phase == 'train':
                net.train()
            else:
                net.eval()

            for batch_idx, batch_data in enumerate(dataloaders[phase]):
                input, target = batch_data['input'].to(device), batch_data['gt'].to(device)
                __, batch_loss = run_batch(net, input, phase, target, l1_weight, criterion, optimizer)

                if batch_idx % 1 == 0:
                    print(phase + ': Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * input.shape[0], len(dataloaders[phase].dataset),
                               100. * batch_idx / len(dataloaders[phase]), batch_loss))
                epoch_loss += batch_loss
            # update loss values
            epoch_loss /= len(dataloaders[phase])
            if phase == 'train':
                train_losses.append(epoch_loss)
            else:
                val_losses.append(epoch_loss)

                # save best model
                if epoch_loss < best_val_loss:
                    best_net = net.state_dict().copy()
                    # torch.save(net.state_dict(), os.path.join(args.model_dir, args.model_path))
                    best_val_loss = epoch_loss
                    bad_epochs = 0
                else:
                    bad_epochs += 1

                # reduce learning rate on plateau
                if lr_reduce:
                    lr_scheduler.step(epoch_loss)

                # early stopping
                if bad_epochs > early_stopping_patience:
                    print("Early stopping after " + str(epoch) + ' epochs')
                    stop = 1
        # visualize loss, prediction and weights of first layer
        if stop:
            break
        if show_training:
            if epoch == 1:
                fig, ax, Lines, weights_to_keep = vis_utils.us_train_vis(dataloaders, epoch, train_losses, val_losses, net, save_training, show_training, res_path,device)
            else:
                fig, ax, Lines, weights_to_keep = vis_utils.us_train_vis(dataloaders, epoch, train_losses, val_losses, net, save_training, show_training, res_path,device, fig, ax, Lines, weights_to_keep)
    if save_training or show_training:
        if not show_training:
            fig, ax, Lines, weights_to_keep = vis_utils.us_train_vis(dataloaders, epoch, train_losses, val_losses, net, save_training, show_training, res_path,device)
        plt.close()
    # torch.save(best_net, model_path+'temp')
    if save_losses:
        best_net['train_loss'] = train_losses
        best_net['val_loss'] = val_losses
    return best_net
