import torch
import torch.nn as nn
import numpy as np
import random
import floris.tools as wfct
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import lr_scheduler

__author__ = "Jens Bauer"
__copyright__ = "Copyright 2021, CNNwake"
__credits__ = ["Jens Bauer"]
__license__ = "MIT"
__version__ = "1.0"
__email__ = "jens.bauer20@imperial.ac.uk"
__status__ = "Development"


class FCNN(nn.Module):
    """
    The class is the Neural Network that can predicts the power output of
    wind turbine and the turbulent intensity (TI) at the turbine. The same
    network architecture is used for both TI and power predict which
    simplifies the code. The network uses the pytorch framwork and uses fully
    connected layers. The methods of this class include the training of
    the network, testing of the accuracy and generaton of training data.
    The networks can be fine tuned via transfer learing if a specific park
    layout is known, this will stongly improve the accuracy.
    """

    def __init__(self, in_size, nr_neurons, out_size=1):
        """
        init method that generates the network architecture using pytroch.
        The number of input varibles can be changed incase more flow data is
        available in the line segment upstream the turbine.
        The nr_neurons defines the size of the given network. The output size
        is set to  1 because the network only predicts either the power or TI.
        In theory it should be able to do both the error was the high
        therefore two networks are used.

        Args:
            in_size (int): Nr. of inputs, usually 42, 40 for wind speed
                and the global ti and yaw angle of the turbine
            nr_neurons (int): Nr. of neurons used in the layers, more
                neurons means that
            the network will have more parameters
                out_size (int): Nr. of outputs in the last layer,
            set to one if the NN only predicts a single value.
        """
        super(FCNN, self).__init__()
        # This defines the model architecture
        self.disc = nn.Sequential(
            # The linear layer is the fully connected layer
            torch.nn.Linear(in_size, nr_neurons),
            # LeakyReLU activation function after every fully
            # connected layer
            torch.nn.LeakyReLU(negative_slope=0.01),
            torch.nn.Linear(nr_neurons, nr_neurons),
            torch.nn.LeakyReLU(negative_slope=0.01),
            torch.nn.Linear(nr_neurons, nr_neurons),
            torch.nn.LeakyReLU(negative_slope=0.01),
            torch.nn.Linear(nr_neurons, out_size),
        )

    def forward(self, x):
        """
        Functions defines a forward pass though the network. Can be used for
        a single input or a batch of inputs

        Args:
            x (torch.tensor): input tensor, to be passed through the network

        Returns:
            flow_fields (torch.tensor): Output of network
        """
        # Use the architecture defined above for a forward pass
        return self.disc(x)

    def initialize_weights(self):
        """
        Initilize weights using a xavier uniform distribution which has
        helped training.
        Loop over all modules, if module is a linear layer then
        initialize weigths.
        For more information about xavier initialization please read:
        Understanding the difficulty of training deep feedforward neural
        networks.
        X. Glorot, und Y. Bengio. AISTATS , Volume 9 JMLR Proceedings,
        249-256, 2010
        """
        # for ever layer in model
        if type(self) == nn.Linear:
            # initialize weights using a xavier distribution
            torch.nn.init.xavier_uniform(self.weight)
            # initialize bias with 0.0001
            self.bias.data.fill_(0.0001)

    @staticmethod
    def power_ti_from_FLORIS(x_position, y_position, yawn_angles,
                             wind_velocity, turbulent_int,
                             type='ti', nr_varabiles=40,
                             florisjason_path='.'):
        """
        This function uses FLORIS to create the dataset to train the FCNN.
        The wind speed along a line just upstream every wind turbine and
        the corresponding TI or power output will be returned as numpy arrays.

        Args:
            x_position (list or numpy array): 1d array of the x postions of
                the wind turbines in m.
            y_position (list or numpy array): 1d array of the y postions of
                the wind turbines in m.
            yawn_angles (lisr numpy array): 1d array of the yaw angle of every
                wind turbinein degree, from -30° to 30°
            wind_velocity (float): Free stream wind velocity in m/s,
                from 3 m/s to 12 m/s
            turbulent_int (float): Turbulent intensity in percent ,
                from 1.5% to 25%
            type (str): Type of data that is returned, if set to power,
                the power generated by every turbine is Returned. If set to
                anything else, the func will return the TI
            nr_varabiles (int): Nr of points along the line upstream the
             turbine to take u values from. More points means that more wind
             speeds are sampled from upstream the turbine. 40 was a good value
            florisjason_path (string): Location of the FLORIS jason file

        Returns:
            numpy array: Final 2d array of flow field around the wind park.

            U_list (2d np.array): array of size len(x_position) x 1 x
             nr_varabiles + 2 where all the wind speeds upstream every
             turbine are stored
            ti_power_list (np.array): array of size len(x_position) x 1
            where either all power or TI values of the turbines are stored
        """

        # define the x and y length of a single cell in the array
        # This is set by the standard value used in FLROIS wakes
        dx = 18.4049079755
        dy = 2.45398773006
        # Set the maximum length of the array to be 3000m and 400m
        # more than the maximum x and y position of the turbines
        x_max = np.max(x_position) + 3005
        y_max = np.max(y_position) + 400
        # Number of cells in x and y needed to create a 2d array of
        # the maximum size
        Nx = int(x_max / dx)
        Ny = int(y_max / dy)

        # Init FLORIS from the jason file
        wind_farm = wfct.floris_interface.FlorisInterface("FLORIS_input"
                                                          "_gauss.json")

        # Set the x and y postions of the wind turbines
        wind_farm.reinitialize_flow_field(layout_array=[x_position,
                                                        y_position])
        # Set the yaw angle of every turbine
        for _ in range(0, len(x_position)):
            wind_farm.change_turbine([_], {'yaw_angle': yawn_angles[_],
                                           "blade_pitch": 0.0})

        # Set inlet wind speed and TI
        wind_farm.reinitialize_flow_field(wind_speed=wind_velocity,
                                          turbulence_intensity=turbulent_int)
        # Calculate wind field
        wind_farm.calculate_wake()

        # Extract 2d slice from 3d domain at hub height
        # This slice needs to have the same number of cells in x and y
        # and same physical dimensions
        cut_plane = wind_farm.get_hor_plane(
            height=90, x_resolution=Nx, y_resolution=Ny, x_bounds=[0, x_max],
            y_bounds=[0, y_max]).df.u.values.reshape(Ny, Nx)

        # Calculate power generated by every turbine
        power = wind_farm.get_turbine_power()
        # Calculate local TI at every tribune
        ti = wind_farm.get_turbine_ti()

        # Initialize list to store all all the u values
        # Number of turbines x 1 x number of values used + 2
        U_list = np.zeros((len(x_position), 1, nr_varabiles + 2))
        # Initialise list to store TI or u valurs
        ti_power_list = np.zeros((len(x_position), 1))

        # From the flow field generated by FLORIS, extract the wind speeds
        # from a line 60 meter upstream the turbines
        for i in range(len(x_position)):
            # determine the x and y cells that the tubine center is at
            turbine_cell = [int((x_position[i]) / dx),
                            int((y_position[i] - 200) / dy)]

            # extract wind speeds along the rotor, 60 meters upstream
            u_upstream_hub = cut_plane[
                             turbine_cell[1] + 45: turbine_cell[1] + 110,
                             turbine_cell[0] - 3]
            # Do an running average, this is done because CNNwake has slight
            # variations in the u predictions, also normalise the u values
            u_average = [((u_upstream_hub[i - 1] +
                           u_upstream_hub[i] +
                           u_upstream_hub[i + 1]) / 3) / 12 for i in
                         np.linspace(1, 63, nr_varabiles, dtype=int)]
            # append yaw which is normalised and ti
            u_average = np.append(u_average, yawn_angles[i] / 30)
            u_input_fcnn = np.append(u_average, turbulent_int)

            U_list[i] = u_input_fcnn

            # If type required is power then use power else
            # use TI
            if type == 'power':
                ti_power_list[i] = power[i]
            else:
                ti_power_list[i] = ti[i]

        # round values to 2 places
        return np.round(U_list, 2), np.round(ti_power_list, 2)

    @staticmethod
    def create_ti_power_dataset(size, u_range, ti_range, yaw_range,
                                nr_varabiles=40, type='power',
                                floris_path='.'):
        """
        This function will create a training or test set to train the power
        or turbulent intensity (TI) prediction networks. The function will
        use FLORIS to create the flowfield around 4 example wind parks
        and saves the wind speed just upstream the wind rotor of every turbine
        and the corresponding TI or power output. The wind speeds are along a
        line which spans the entire diameter of the turbine blades and along
        this line nr_varibles of points are sampled and the wind farm TI and
        yaw angle of the corresponding turbine is added.
        This allows the network to predict the power output of every turbine
        under different inflow conditions or TI at every trubine.
        Four different wind parks examples are used to generate the data,
        this does not cover all possible flow fields
        but delivers a good inital guess for the network.
        The corresponding TI or power values are normalised by the maximum
        value of the array, this will make all values to be between
        0 and 1 which helps training.

        Args:
            size (int, optional): Nr of example flows generated and saved for
                training. Defaults to 400.
            u_range (list): Bound of u values [u_min, u_max] used
            ti_range (list): Bound of TI values [TI_min, TI_max] used
            yaw_range (list): Bound of yaw angles [yaw_min, yaw_max] used
            nr_varabiles (int, optional): Nr. of values sampled along line.
                Defaults to 40.
            type (str, optional): If set to power, the power will be saved,
                if set to anything else the TI at every turbine will be saved
                Defaults to 'power'.
            floris_path (str, optinal): Path to FLORIS jason file.

        Returns:
            x [torch tensor]: Tensor of size size*6 x 1 x nr_varabiles+2 where
             all the flow data along line is stored. This will be the input
             to the FCNN
            y [torch tensor]: Tensor of size chuck_size*6 x 1 where all the
             TI or pwoer data for every turbine is stored, this is what the
             FCNN is trained to predict
        """

        # 4 wind parks are used the generate data
        # for every wind park generates 1/4 of the dataset
        chuck_size = int(size/4)

        # initialize empty numpy array to store 2d arrays
        # and corresponding u, ti and yawn values
        y = np.zeros((chuck_size * 4 * 6, 1, nr_varabiles + 2))
        x = np.zeros((chuck_size * 6 * 4, 1))

        # index to add the wind fields in the right postion
        index = [i for i in range(0, size * 6, 6)]

        # create train examples
        print("generate FLORIS data")

        # WIND PARK 1
        for _ in range(0, chuck_size):
            # sample u, ti and yaw from uniform distro
            u_list = round(random.uniform(u_range[0], u_range[1]), 2)
            ti_list = round(random.uniform(ti_range[0], ti_range[1]), 2)
            yawlist = [round(random.uniform(yaw_range[0], yaw_range[1]), 2) for _ in range(0, 6)]

            # get the wind speeds along line and corresponding TI or power
            # from FLORIS for the wind park
            u_list_hub, floris_power_ti = FCNN.power_ti_from_FLORIS(
                [100, 300, 1000, 1300, 2000, 2300],
                [300, 500, 300, 500, 300, 500],
                yawlist, u_list, ti_list, type, nr_varabiles,
                florisjason_path=floris_path)

            # add u and power/TI in correct postion
            y[index[_]: index[_ + 1], :, :] = u_list_hub
            x[index[_]: index[_ + 1], :] = floris_power_ti

        # WIND PARK 2
        for _ in range(chuck_size, chuck_size * 2):
            u_list = round(random.uniform(u_range[0], u_range[1]), 2)
            ti_list = round(random.uniform(ti_range[0], ti_range[1]), 2)
            yawlist = [round(random.uniform(yaw_range[0], yaw_range[1]), 2) for _ in range(0, 6)]

            u_list_hub, floris_power_ti = FCNN.power_ti_from_FLORIS(
                [100, 600, 1000, 1300, 2000, 2900],
                [300, 300, 300, 300, 300, 500],
                yawlist, u_list, ti_list,  type, nr_varabiles)

            y[index[_]: index[_ + 1], :, :] = u_list_hub
            x[index[_]: index[_ + 1], :] = floris_power_ti

        # WIND PARK 3
        for _ in range(chuck_size * 2, chuck_size * 3):
            u_list = round(random.uniform(u_range[0], u_range[1]), 2)
            ti_list = round(random.uniform(ti_range[0], ti_range[1]), 2)
            yawlist = [round(random.uniform(yaw_range[0], yaw_range[1]), 2) for _ in range(0, 6)]

            u_list_hub, floris_power_ti = FCNN.power_ti_from_FLORIS(
                [100, 100, 800, 1600, 1600, 2600],
                [300, 500, 400, 300, 500, 400],
                yawlist, u_list, ti_list,  type, nr_varabiles)

            y[index[_]: index[_ + 1], :, :] = u_list_hub
            x[index[_]: index[_ + 1], :] = floris_power_ti

        # WIND PARK 4
        for _ in range(chuck_size * 3, chuck_size * 4 - 1):
            u_list = round(random.uniform(u_range[0], u_range[1]), 2)
            ti_list = round(random.uniform(ti_range[0], ti_range[1]), 2)
            yawlist = [round(random.uniform(yaw_range[0], yaw_range[1]), 2) for _ in range(0, 6)]

            u_list_hub, floris_power_ti = FCNN.power_ti_from_FLORIS(
                [100, 300, 500, 1000, 1300, 1600],
                [300, 500, 300, 300, 500, 400],
                yawlist, u_list, ti_list,  type, nr_varabiles)

            y[index[_]: index[_ + 1], :, :] = u_list_hub
            x[index[_]: index[_ + 1], :] = floris_power_ti

        # transform into a pytroch tensor
        x = torch.tensor(x[0:-6], dtype=torch.float)
        y = torch.tensor(y[0:-6], dtype=torch.float)

        print(f"Normalisation used: {torch.max(x)}")
        # Normalise the power/TI by maximum value so that they are
        #  between 0-1
        x = x / torch.max(x)

        return y, x

    def epoch_training(self, criterion, optimizer, dataloader, device):
        """
        Trains the model for one epoch data provided by dataloader. The model
        will be updated after each batch and the function will return the
        train loss of the last batch

        Args:
            criterion (torch.nn.criterion): Loss function used to
                train model
            optimizer (torch.optim.Optimizer): Optimizer used for
                gradient descent
            dataloader (torch.utils.data.DataLoader): Dataloader for dataset
            device (str): Device on which model and data is stored,
                cpu or cuda

        Returns:
            training loss (float): Loss value of training set defined
             by criterion
        """
        # For all data in datalaoder
        for power_ti, input_u in dataloader:
            # one batch at a time, get network prediction
            output = self(input_u.to(device))
            # compute loss
            train_loss = criterion(output.squeeze(), power_ti[:, 0].to(device))

            self.zero_grad()  # Zero the gradients
            train_loss.backward()  # Calc gradients
            optimizer.step()  # Do parameter update

        return train_loss.item()

    def learn_wind_park(self, x_postion, y_position, size, eval_size,
                        nr_varabiles=40, type='power',
                        device='cpu', nr_epochs=50,
                        batch_size=100, lr=0.003):
        """
        EXPERIMENTAL FUNCTION; DOES NOT WORK YET, DO NOT USE!!!
        This function is supposed to fine tune a already trained TI/Power
        model on a specific wind park. This should further reduce the error
        in predicting power or local TI. However, it currently increase the
        error so there is something wrong. DO NOT USE!!!!

        Args:
            x_postion (list or numpy array): 1d array of the x positions of
                the wind turbines in m.
            y_position (list or numpy array): 1d array of the y positions of
                the wind turbines in m.
            size (list numpy array): Size of training set
            eval_size (list numpy array): Size of test set
            nr_varabiles (int): Nr of points along the line upstream the
                turbine to take u values from. More points means that more
                speeds are sampled from upstream the turbine. 40 was a good
            type (str): Type of data that is returned, if set to power,
                the power generated by every turbine is Returned. If set to
                anything else, the func will return the TI
            device (torch.device): Device to run the training on, cuda or cpu
            nr_epochs (int): Nr. of training epochs
            batch_size (int): Training batch size
            lr (float): Model learning rate

        Returns:
            [Bool]: True if training was successful
        """

        nr_values = int(((size + eval_size)*len(x_postion)))

        # initialize empty numpy array to store 2d arrays and
        # corresponding u, ti and yawn values
        y = np.zeros((nr_values, 1, nr_varabiles + 2))
        x = np.zeros((nr_values, 1))

        print(nr_values)
        print(len(x_postion))
        print(int(nr_values/len(x_postion)))

        index = [i for i in range(0, nr_values * 2, len(x_postion))]

        # create train examples of the specified wind farm using FLORIS
        print("generate FLORIS data")
        for _ in range(0, int(nr_values/len(x_postion))):
            u_list = round(random.uniform(3, 12), 2)
            ti_list = round(random.uniform(0.015, 0.25), 2)
            yawlist = [round(random.uniform(-30, 30), 2)
                       for _ in range(0, len(x_postion))]

            u_list_hub, floris_power_ti = FCNN.power_ti_from_FLORIS(
                x_postion, y_position, yawlist, u_list, ti_list, type,
                nr_varabiles)

            y[index[_]: index[_ + 1], :, :] = u_list_hub
            x[index[_]: index[_ + 1], :] = floris_power_ti

        x = torch.tensor(x, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.float)

        print(f"Normalisation used: {torch.max(x)}")
        x = x / torch.max(x)

        x_train = x[0:size * len(x_postion)]
        y_train = y[0:size * len(x_postion)]

        x_eval = x[-eval_size*len(x_postion):]
        y_eval = y[-eval_size*len(x_postion):]

        print(x_eval.size(), x_train.size())

        dataset = TensorDataset(x_train, y_train.float())
        # generate dataload for training
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = optim.Adam(self.parameters(), lr=lr)
        scheduler_gen = lr_scheduler.ReduceLROnPlateau(optimizer, 'min',
                                                       factor=0.6, patience=4,
                                                       verbose=True)

        # use L2 norm as criterion
        criterion = nn.MSELoss()

        # init to list to store error
        error_list = []

        # Train model on data
        for _ in range(nr_epochs):  # train model

            self.train()  # set model to training mode

            loss = self.epoch_training(criterion, optimizer,
                                       dataloader, device)

            self.eval()  # set model to evaluation
            # evaluation on validation set
            val_error = self.error(y_eval, x_eval, device)
            # if error has not decreased over the past 4 epochs
            # decrease the lr by a factor of 0.6
            scheduler_gen.step(val_error)

            error_list.append(val_error)

            print(f" Epoch: {_:.0f}, Training loss: {loss:.4f},"
                  f" Validation error: {val_error:.2f}")

        # plot the val error over the epochs
        plt.plot(range(nr_epochs), error_list)
        plt.show()

        return True

    def error(self, x_eval, y_eval, device='cpu'):
        """
        Function to calculate the error between the networks
        predictions and the actual output. The x and y values
        need to be generated using the create_ti_power_dataset
        function. The error will be the mean percentage difference
        between all values predicted by the network and the actual
        values

        Args:
            x_eval (torch tensor): Tensor of all flow, ti and yaw values
                for different turbines, this the the model input.
            y_eval (torch tensor): Tensor of all TI or power outputs as
                calculated by floris for the corresponding flow field in x
            device (str, optional): Device where the model is stored on.
                Defaults to 'cpu'.

        Returns:
             error (float): percentage error
        """
        error_list = []
        # Do forward pass of the x data
        model_predict = self.forward(x_eval.to(device))
        for n in range(0, len(y_eval)):
            # sometimes the power prediction is zero, this will give
            # an error of inf due to divide by zero in step below.
            # Therefore filter out very small power here
            if abs(y_eval.detach().cpu().numpy()[n]) < 0.01:
                continue
            else:
                # calculate error
                power_error = abs(y_eval.detach().cpu().numpy()[n] -
                                  model_predict[n].detach().cpu().numpy()) / (
                            y_eval.detach().cpu().numpy()[n] + 1e-8)
                error_list.append(power_error * 100)

        return np.mean(error_list)

    def load_model(self, path='.', device='cpu'):
        """
        Function to load model from a pt file into this class.

        Args:
            path (str): path to saved model.
            device (torch.device): Device to load onto, cpu or cuda

        """
        # Load a previously trained model
        self.load_state_dict(torch.load(path, map_location=device))

    def save_model(self, name='generator.pt'):
        """
        Function to save current model paramters so that it can
        be used again later. Needs to be saved with as .pt file

        Args:
            name (str): name of .pt file from which to load model
        """
        torch.save(self.state_dict(), name)
