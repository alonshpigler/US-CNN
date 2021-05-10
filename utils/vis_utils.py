import pandas as pd
import matplotlib.pyplot as plt
import torch
import math
import os
import numpy as np
from scipy.signal import convolve, hilbert


def show_phase_power_rep(signal, q_base, i_base, pow_rep, angle, n_angle, d_angle, n_phase_rep, m_angle, nd_angle):

    linewidth=0.5
    fig, ax = plt.subplots(nrows=3, ncols=4,
                           sharex=False, sharey=False,)
    ax[0, 0].plot(signal[0,:].cpu().numpy(),linewidth=linewidth)
    ax[0, 0].set_title('Signal')
    ax[0, 1].plot(q_base.cpu().numpy(),linewidth=linewidth)
    ax[0, 1].set_title('Q')
    ax[0, 2].plot(i_base.cpu().numpy(),linewidth=linewidth)
    ax[0, 2].set_title('I')
    ax[0, 3].plot(pow_rep.cpu().numpy(),linewidth=linewidth)
    ax[0, 3].set_title('pow')
    ax[1, 0].plot(angle.cpu().numpy(),linewidth=linewidth)
    ax[1, 0].set_title('angle')
    ax[1, 1].plot(n_angle.cpu().numpy(),linewidth=linewidth)
    ax[1, 1].set_title('power*angle')
    ax[1, 2].plot(d_angle.cpu().numpy(),linewidth=linewidth)
    ax[1, 2].set_title('d_ang')
    ax[1, 2].set_ylim([-math.pi, 2 * math.pi])
    ax[1, 3].plot(n_phase_rep.cpu().numpy(),linewidth=linewidth)
    ax[1, 3].set_title('d_ang%2pi')
    ax[1, 3].set_ylim([-math.pi, 2 * math.pi])
    ax[2, 0].plot(m_angle.cpu().numpy(),linewidth=linewidth)
    ax[2, 0].set_title('d_ang%2pi-e')
    ax[2, 0].set_ylim([-math.pi / 2, 3 * math.pi / 4])
    ax[2, 1].plot(nd_angle.cpu().numpy(),linewidth=linewidth)
    ax[2, 1].set_title('(d_ang%2pi-e)*pow')
    ax[2,1].set_ylim([-math.pi/2, 3*math.pi/4])

    fig.tight_layout()
    plt.show()


def us_train_vis(dataloader, epoch, train_losses, test_losses, model, save_training, show_training, res_path,device, fig=[], ax=[], lines={}, weights_to_keep=[],show_weights = True):
    """
    Visualize training after each epoch. Show example, loss and 1st layer weights
    :param epoch:
    :param train_losses:
    :param test_losses:
    :param model:
    :param args:
    :param fig:
    :param ax:
    :param lines:
    :param weights_to_keep:
    :return:
    """

    n_weights_to_show = 3
    N = dataloader['train'].dataset[0]['input'].shape[1]

    train_input_example = dataloader['train'].dataset[0]['input'].squeeze().tolist()
    train_gt_example = dataloader['train'].dataset[0]['gt'].tolist()
    train_output_example = model(dataloader['train'].dataset[0]['input'].reshape([1,1,N,1]).to(device)).tolist()[0][0]

    val_input_example = dataloader['val'].dataset[0]['input'].squeeze().tolist()
    val_gt_example = dataloader['val'].dataset[0]['gt'].tolist()
    val_output_example = model(dataloader['val'].dataset[0]['input'].reshape([1,1,N,1]).to(device)).tolist()[0][0]

    # N = args.DataGeneration['signal_n'] * args.DataGeneration['fs']
    #  Initializing figure
    if epoch ==1 or (save_training and not show_training):
        fig, ax = plt.subplots(nrows=4, ncols=2, figsize=(12, 7),
                               sharex=False, sharey=False)
        fig.suptitle('After ' + str(epoch) + ' epochs')

        # Plot sample of  train data
        lines ={
            'y_signal_train': ax[0, 0].plot(train_input_example, color='b', label='input signal'),
            'x_delta_train': ax[0, 0].plot(train_gt_example, color='g', label='ground truth')
        }
        ax[0, 0].set_title('Train - Data and gt example')
        ax[0, 0].legend(loc="upper right")

        # Plot prediction of train data sample
        ax[1,0].plot(train_gt_example, color='g', label='ground truth')
        lines['x_delta_train_pred'], = ax[1, 0].plot(train_output_example,label='Prediction')
        ax[1,0].legend(loc="lower right")
        ax[1,0].set_title('Train - Prediction example')

        # Plot sample of  val data
        lines['y_signal_val'], = ax[0, 1].plot(val_input_example, color='b', label='input signal'),
        lines['x_delta_val'] = ax[0, 1].plot(val_gt_example, color='g', label='ground truth')
        ax[0, 1].set_title('Validation data example')
        ax[0, 1].legend(loc="upper right")

        # Plot prediction of data sample
        ax[1, 1].plot(val_gt_example, color='g', label='ground truth')
        lines['x_delta_val_pred'], = ax[1, 1].plot(val_output_example, label='Prediction')
        ax[1, 1].legend(loc="lower right")
        ax[1, 1].set_title('Validation prediction example')

        # Plot train and test loss
        lines['train_loss'], = ax[2, 0].plot(range(1, epoch + 1), train_losses, label='train loss')
        lines['test_loss'], = ax[2, 0].plot(range(1, epoch + 1), test_losses, label='test loss')
        ax[2, 0].legend(loc="upper right")
        ax[2, 0].set_title('Loss')

        # Plot weights of first conv layer during training
        if show_weights:
            weights = model.down_path[0].weight.data
            for c in range(n_weights_to_show):
                weights_to_keep.append(ax[math.ceil((c + 4) / 2), (c + 1) % 2].plot(
                    weights[c,0,:,0].cpu().numpy(),
                    color='b', label='model weights'))
                ax[math.ceil((c + 4) / 2), (c + 1) % 2].set_title('model weights ' + str(c + 1))
                ax[math.ceil((c + 4) / 2), (c + 1) % 2].legend(loc="upper right")

        plt.tight_layout()

    else:
        # Update train and test loss
        fig.suptitle('After ' + str(epoch) + ' epochs')
        lines['x_delta_train_pred'].set_ydata(train_output_example)
        lines['x_delta_val_pred'].set_ydata(val_output_example)
        lines['train_loss'].set_xdata(range(1, epoch + 1))
        lines['train_loss'].set_ydata(train_losses)
        lines['test_loss'].set_xdata(range(1, epoch + 1))
        lines['test_loss'].set_ydata(test_losses)
        ax[2, 0].set_xlim(1, epoch)
        ax[2, 0].set_ylim(0, np.average(train_losses))

        # Update weights visualization
        weights = model.down_path[0].weight.data
        for c in range(n_weights_to_show):
            weights_to_keep[c][0].set_ydata(
                # weights.tolist()[(c - 1) * args.net_args['big_filter_width']:(c * args.net_args['big_filter_width'] - 1)])
                weights[c, 0, :, 0].cpu().numpy()
            )
            ax[math.ceil((c + 4) / 2), (c + 1) % 2].set_ylim(
                # min(weights.tolist()[(c - 1) * args.net_args['big_filter_width']:(c * args.net_args['big_filter_width'] - 1)]))
                bottom=min(weights[c, 0, :, 0].cpu().numpy()), top= max(weights[c, 0, :, 0].cpu().numpy())
            )
    plt.pause(.00001)
    if save_training:
        plt.savefig(os.path.join(res_path, 'training-vis.jpg'))

    return fig, ax, lines, weights_to_keep


def nice_plot(lines, labels=[], show=1, save=0, save_path='.'):
    """
    :param lines: list of lines to plot, each type ndarray or list
    :param labels: labels for the lines. optional
    :param show: if 1, plot is shown onto screen. Default: 1
    :param save: if 1, plot is saved to file (and save_path needs to be defined as well. Default: 0
    :param save_path: necessary if save==1
    """

    for i in range(len(lines)):
        if len(labels) > i:
            plt.plot(lines[i], label = labels[i])
        else:
            plt.plot(lines[i])

    plt.legend()
    if show:
        plt.show()
    if save:
        plt.savefig(save_path)

