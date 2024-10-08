#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    ANFIS in torch: some simple functions to supply data and plot results.
    @author: James Power <james.power@mu.ie> Apr 12 18:13:10 2019
"""

import matplotlib.pyplot as plt
import logging
import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, filename='./training_log.log', filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')

dtype = torch.float

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    """
    Saves a training checkpoint.

    Parameters:
    - state: (dict) training state, including:
               - 'epoch': the current epoch (int)
               - 'model_state_dict': the model's state dict (dict)
               - 'optimizer_state_dict': the optimizer's state dict (dict)
    - filename: (str) where to save the checkpoint (default: 'checkpoint.pth.tar')
    """
    torch.save(state, filename)
def load_checkpoint(filename, model, optimizer=None):
    """
    Loads a training checkpoint.

    Parameters:
    - filename: (str) the path to the checkpoint file
    - model: the model to load the state dict into
    - optimizer: (optional) the optimizer to load the state dict into
    """
    checkpoint = torch.load(filename)
    epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model, optimizer, epoch



class TwoLayerNet(torch.nn.Module):
    '''
        From the pytorch examples, a simjple 2-layer neural net.
        https://pytorch.org/tutorials/beginner/pytorch_with_examples.html
    '''
    def __init__(self, d_in, hidden_size, d_out):
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(d_in, hidden_size)
        self.linear2 = torch.nn.Linear(hidden_size, d_out)

    def forward(self, x):
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred


def linear_model(x, y, epochs=200, hidden_size=10):
    '''
        Predict y from x using a simple linear model with one hidden layer.
        https://pytorch.org/tutorials/beginner/pytorch_with_examples.html
    '''
    assert x.shape[0] == y.shape[0], 'x and y have different batch sizes'
    d_in = x.shape[1]
    d_out = y.shape[1]
    model = TwoLayerNet(d_in, hidden_size, d_out)
    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    errors = []
    for t in range(epochs):
        y_pred = model(x)
        tot_loss = criterion(y_pred, y)
        perc_loss = 100. * torch.sqrt(tot_loss).item() / y.sum()
        errors.append(perc_loss)
        if t % 10 == 0 or epochs < 20:
            print('epoch {:4d}: {:.5f} {:.2f}%'.format(t, tot_loss, perc_loss))
        optimizer.zero_grad()
        tot_loss.backward()
        optimizer.step()
    return model, errors


def plotErrors(errors):
    '''
        Plot the given list of error rates against no. of epochs
    '''
    plt.plot(range(len(errors)), errors, '-ro', label='errors')
    plt.ylabel('Percentage error')
    plt.xlabel('Epoch')
    plt.show()


def plotResults(y_actual, y_predicted):
    '''
        Plot the actual and predicted y values (in different colours).
    '''
    plt.plot(range(len(y_predicted)), y_predicted.detach().numpy(),
             'r', label='trained')
    plt.plot(range(len(y_actual)), y_actual.numpy(), 'b', label='original')
    plt.legend(loc='upper left')
    plt.show()


def _plot_mfs(var_name, fv, x):
    '''
        A simple utility function to plot the MFs for a variable.
        Supply the variable name, MFs and a set of x values to plot.
    '''
    # Sort x so we only plot each x-value once:
    xsort, _ = x.sort()
    for mfname, yvals in fv.fuzzify(xsort):
        plt.plot(xsort.tolist(), yvals.tolist(), label=mfname)
    plt.xlabel('Values for variable {} ({} MFs)'.format(var_name, fv.num_mfs))
    plt.ylabel('Membership')
    plt.legend(bbox_to_anchor=(1., 0.95))
    plt.show()


def plot_all_mfs(model, x):
    for i, (var_name, fv) in enumerate(model.layer.fuzzify.varmfs.items()):
        _plot_mfs(var_name, fv, x[:, i])


def calc_error(y_pred, y_actual):
    with torch.no_grad():
        tot_loss = F.mse_loss(y_pred, y_actual)
        rmse = torch.sqrt(tot_loss).item()
        # perc_loss = torch.mean(100. * torch.abs((y_pred - y_actual)
        #                        / y_actual))
        perc_loss =F.l1_loss(y_pred, y_actual, reduction='mean')
    return(tot_loss, rmse, perc_loss)


def test_anfis(model, data, show_plots=False):
    '''
        Do a single forward pass with x and compare with y_actual.
    '''
    x, y_actual = data.dataset.tensors
    if show_plots:
        plot_all_mfs(model, x)
    print('### Testing for {} cases'.format(x.shape[0]))
    y_pred = model(x)
    mse, rmse, perc_loss = calc_error(y_pred, y_actual)
    print('MS error={:.5f}, RMS error={:.5f}, percentage={:.2f}%'
          .format(mse, rmse, perc_loss))
    if show_plots:
        plotResults(y_actual, y_pred)


def train_anfis_with(model, data, optimizer, criterion,
                     epochs=500, show_plots=False):
    '''
        Train the given model using the given (x,y) data.
    '''
    errors = []  # Keep a list of these for plotting afterwards
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    print('### Training for {} epochs, training size = {} cases'.
          format(epochs, data.dataset.tensors[0].shape[0]))
    for t in range(epochs):
        # Process each mini-batch in turn:
        for x, y_actual in data:
            y_pred = model(x)
            # Compute and print loss
            loss = criterion(y_pred, y_actual)
            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # Epoch ending, so now fit the coefficients based on all data:
        x, y_actual = data.dataset.tensors
        with torch.no_grad():
            model.fit_coeff(x, y_actual)
        # Get the error rate for the whole batch:
        y_pred = model(x)
        mse, rmse, perc_loss = calc_error(y_pred, y_actual)
        errors.append(perc_loss)
        # Print some progress information as the net is trained:
        if epochs < 30 or t % 10 == 0:
            print('epoch {:4d}: MSE={:.5f}, RMSE={:.5f} MAE={:.2f}%'
                  .format(t, mse, rmse, perc_loss))
    # End of training, so graph the results:
    if show_plots:
        plotErrors(errors)
        y_actual = data.dataset.tensors[1]
        y_pred = model(data.dataset.tensors[0])
        plotResults(y_actual, y_pred)


def train_anfis(model, data, epochs=500, show_plots=False):
    '''
        Train the given model using the given (x,y) data.
    '''
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.99)
    criterion = torch.nn.MSELoss(reduction='sum')
    train_anfis_with(model, data, optimizer, criterion, epochs, show_plots)

def train_anfis_with_dev_data(model, data,dev_data, optimizer, criterion,
                     epochs=500, show_plots=False):
    logging.basicConfig(level=logging.INFO, filename='./training_log.log', filemode='a',
                        format='%(asctime)s - %(levelname)s - %(message)s')
    '''
        Train the given model using the given (x,y) data.
    '''
    errors = []  # Keep a list of these for plotting afterwards
    errors_dev = []
    best_error = float('inf')
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    print('### Training for {} epochs, training size = {} cases'.
          format(epochs, data.dataset.tensors[0].shape[0]))


    for t in range(epochs):
        # Process each mini-batch in turn:
        for x, y_actual in data:
            y_pred = model(x)
            # Compute and print loss
            loss = criterion(y_pred, y_actual)
            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # Epoch ending, so now fit the coefficients based on all data:
        x, y_actual = data.dataset.tensors
        with torch.no_grad():
            model.fit_coeff(x, y_actual)
        # Get the error rate for the whole batch:
        y_pred = model(x)
        mse, rmse, perc_loss = calc_error(y_pred, y_actual)
        errors.append(perc_loss)
        # Print some progress information as the net is trained:
        # if epochs < 30 or t % 10 == 0:
        #     print('epoch {:4d}: train_MSE={:.5f}, train_RMSE={:.5f} ={:.2f}%'
        #           .format(t, mse, rmse, perc_loss))



        with torch.no_grad():
            model.eval()
            devloss = 0
            for x_dev,y_dev in dev_data:
                y_pred_dev = model(x_dev)
                devloss+= criterion(y_pred_dev,y_dev).item()
            devloss /=len(dev_data)


        # Get the error rate for the whole batch:
        y_pred = model(x)
        mse, rmse, perc_loss = calc_error(y_pred, y_actual)
        errors.append(perc_loss)

        y_pred_dev = model(x_dev)
        mse_dev,rmse_dev,perc_loss_dev = calc_error(y_pred_dev,y_dev)
        errors_dev.append(perc_loss_dev)
        # Print some progress information as the net is trained:
        if epochs < 30 or t % 10 == 0:
            print('epoch {:4d}: train_MSE={:.5f}, train_RMSE={:.5f} MAE={:.2f}%'
                  .format(t, mse, rmse, perc_loss))
            print('epoch {:4d}: dev_MSE={:.5f}, dev_RMSE={:.5f} MAE={:.2f}%'
                  .format(t, mse_dev, rmse_dev, perc_loss_dev))

            logging.info(f'epoch {t:4d}: train_MSE={mse:.5f}, train_RMSE={rmse:.5f}, MAE={perc_loss:.2f}%')
            # 记录验证指标
            logging.info(f'epoch {t:4d}: dev_MSE={mse_dev:.5f}, dev_RMSE={rmse_dev:.5f}, MAE={perc_loss_dev:.2f}%')

        checkpoint = {
            'epoch': t + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        save_checkpoint(checkpoint, filename=f'checkpoint_epoch_{t + 1}.pth.tar')


        # best check point
        error = rmse_dev
        if error<best_error:
            best_error = error
            best_checkpoint = {
                'epoch': t + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'error':best_error,
            }
            save_checkpoint(best_checkpoint,filename='best_checkpoint.pth.tar')
    # End of training, so graph the results:
    if show_plots:
        plotErrors(errors)
        y_actual = data.dataset.tensors[1]
        y_pred = model(data.dataset.tensors[0])
        plotResults(y_actual, y_pred)



if __name__ == '__main__':
    x = torch.arange(1, 100, dtype=dtype).unsqueeze(1)
    y = torch.pow(x, 3)
    model, errors = linear_model(x, y, 1000)
    plotErrors(errors)
    plotResults(y, model(x))
