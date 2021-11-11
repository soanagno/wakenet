import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import lr_scheduler
from FCC_model import FCNN


__author__ = "Jens Bauer"
__copyright__ = "Copyright 2021, CNNwake"
__credits__ = ["Jens Bauer"]
__license__ = "MIT"
__version__ = "1.0"
__email__ = "jens.bauer20@imperial.ac.uk"
__status__ = "Development"


def train_FCNN_model(
        nr_neurons, input_size, nr_epochs, learing_rate, batch_size,
        train_size, val_size, u_range, ti_range, yaw_range, model_name,
        type='power', device='cpu', nr_workers=0, floris_path="."):
    """
    Create a new model and train it for a certain number of epochs using a
    newly generated dataset. Hyper-parameters such as model size or lr can be
    changed as input to the function.
    After training the model error over all epochs is plotted and the model
    performance will be evaluated on a unseen test set. Finally, the model
    will saved as the model_name which needs to add as .pt file

    Args:
        nr_filters (int): Nr. of filters used for the conv layers
        nr_epochs (int): Nr. of training epochs
        learing_rate (float): Model learing rate
        batch_size (int): Training batch size
        train_size (int): Size of the generated training set
        val_size (int): Size of the generated validation set
        u_range (list): Bound of u values [u_min, u_max] used
        ti_range (list): Bound of TI values [TI_min, TI_max] used
        yaw_range (list): Bound of yaw angles [yaw_min, yaw_max] used
        model_name (str): Name of the trained saved model (needs to be .pt)

        image_size (int): Size of the data set images, needs to match the
            model output size for the current model this is 163
        device (torch.device): Device to run the training on, cuda or cpu
        nr_workers (int, optional): Nr. of workers to load data. Defaults to 0.
        floris_path (str, optinal): Path to FLORIS jason file.

    Returns:
        gen (Generator): Trained model
        loss (float): training loss defined by the loss function
        val_error (float): Percentage error on the validation set
    """

    # The current inputs are: u, ti and yaw. If more are used please
    # change this input var
    model_input_size = input_size + 2

    # create a generator of the specified size
    model = FCNN(model_input_size, nr_neurons, 1).to(device)

    # create a datasets from the data generated by FLORIS
    x_train, y_train = model.create_ti_power_dataset(
        size=train_size, u_range=u_range, ti_range=ti_range,
        yaw_range=yaw_range, nr_varabiles=input_size, type=type,
        floris_path=floris_path)
    x_eval, y_eval = model.create_ti_power_dataset(
        size=val_size, u_range=u_range, ti_range=ti_range,
        yaw_range=yaw_range, nr_varabiles=input_size, type=type,
        floris_path=floris_path)
    dataset = TensorDataset(y_train, x_train.float())
    # generate dataload for training
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=nr_workers)

    # init the weights of the generator
    model.initialize_weights()
    # set up and optimizer and learing rate scheduler using hyperparameters
    optimizer = optim.Adam(model.parameters(), lr=learing_rate)
    scheduler_gen = lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', factor=0.6, patience=4, verbose=True)

    # use L2 norm as criterion
    criterion = nn.MSELoss()

    # init to list to store error
    error_list = []

    for _ in range(nr_epochs):  # train model

        model.train()  # set model to training mode

        loss = model.epoch_training(criterion, optimizer, dataloader, device)

        model.eval()  # set model to evaluation
        # evaluation on validation set
        val_error = model.error(x_eval, y_eval, device)
        # if error has not decreased over the past 4 epochs decrease
        # the lr by a factor of 0.6
        scheduler_gen.step(val_error)

        error_list.append(val_error)

        print(f" Epoch: {_:.0f},"
              f" Training loss: {loss:.4f},"
              f" Validation error: {val_error:.2f}")

    # save model
    model.save_model(model_name)

    # plot the val error over the epochs
    plt.plot(range(nr_epochs), error_list)
    plt.show()

    return model, loss, val_error

if __name__ == '__main__':
    # To run indivual CNNWake files, the imports are not allowed to be
    # relative. Instead of: from .FCC_model import FCNN
    # it needs to be: from FCC_model import FCNN, for all CNNWake imports

    # Set device used for training
    devices = torch.device("cpu" if torch.cuda.is_available() else "cpu")
    # Train a FCNN to predict power
    train_FCNN_model(
        nr_neurons=20, input_size=20, nr_epochs=150, learing_rate=0.003,
        batch_size=30, train_size=50, val_size=40, u_range=[3, 12],
        ti_range=[0.015, 0.25], yaw_range=[-30, 30],
        model_name='power_model.pt', type='power', device=devices)
