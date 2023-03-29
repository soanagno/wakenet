import torch
import torch.nn as nn
import numpy as np
import random
import floris.tools as wfct

__author__ = "Jens Bauer"
__copyright__ = "Copyright 2021, CNNwake"
__credits__ = ["Jens Bauer"]
__license__ = "MIT"
__version__ = "1.0"
__email__ = "jens.bauer20@imperial.ac.uk"
__status__ = "Development"


class Generator(nn.Module):
    """
    The class is the Neural Network that generates the flow field around a
    wind turbine. The network uses the pytorch framwork and uses fully
    connected and transpose convolutional layers.
    The methods of this class include the training of the network,
    testing of the accuracy and generaton of the training data.
    """

    def __init__(self, nr_input_var, nr_filter):
        """
        init method that generates the network architecture using pytroch's
        ConvTranspose2d and Sequential layers. The number of input varibles
        and size of the given network can be changed. The output size will not
        change and it set at 163 x 163 pixels.

        Args:
            nr_input_var (int): Nr. of inputs, usually 3 for u, ti and yaw
            nr_filter (int): Nr. filters used in deconv layers, more filters
                             means that the network will have more parameters
        """
        super(Generator, self).__init__()
        # linear layer
        self.FC_Layer = nn.Sequential(nn.Linear(in_features=nr_input_var,
                                                out_features=9))
        # Deconvolutional layer
        self.net = nn.Sequential(
            self.layer(1, nr_filter * 16, 4, 2, 1),
            self.layer(nr_filter * 16, nr_filter * 8, 4, 1, 1),
            self.layer(nr_filter * 8, nr_filter * 8, 4, 2, 1),
            self.layer(nr_filter * 8, nr_filter * 4, 4, 2, 1),
            self.layer(nr_filter * 4, nr_filter * 4, 3, 2, 1),
            nn.ConvTranspose2d(nr_filter * 4, 1, kernel_size=3,
                               stride=3, padding=1),
        )

    def layer(self, in_filters, out_filters, kernel_size, stride, padding):
        """
        One layer of the CNN which consits of ConvTranspose2d,
        a batchnorm and LRelu activation function.
        Function is used to define one layer of the network

        Args:
            in_filters (int): Nr. of filters in the previous layer
            out_filters (int): Nr. of output filters
            kernel_size (int): Size of the ConvTranspose2d layer
            stride (int): Stride of the ConvTranspose2d layer
            padding (int): Padding used in this layer

        Returns:
            nn.Sequential: Pytroch Sequential container that defines one layer
        """
        # One layer of the network uses:
        # Deconvolutional layer, then batch norm and leakyrelu
        # activation function
        single_layer = nn.Sequential(nn.ConvTranspose2d(in_filters,
                                                        out_filters,
                                                        kernel_size,
                                                        stride,
                                                        padding,
                                                        bias=False, ),
                                     nn.BatchNorm2d(out_filters),
                                     nn.LeakyReLU(0.2), )

        return single_layer

    def initialize_weights(self):
        """
        Initilize weights using a normal distribution with mean = 0,std2 = 0.02
        which has helped training. Loop over all modules, if module is
        convolutional layer or batchNorm then initialize weights.

        Args:
            model (torch model): Neural network model defined using Pytorch
        """
        # for ever layer in model
        for m in self.modules():
            # check if it deconvolutional ot batch nrom layer
            if isinstance(m, (nn.Conv2d, nn.BatchNorm2d)):
                # initialize weights using a normal distribution
                nn.init.normal_(m.weight.data, 0.0, 0.02)

    def forward(self, x):
        """
        Functions defines a forward pass though the network. Can be used for
        a single input or a batch of inputs

        Args:
            x (torch.tensor): input tensor, to be passed through the network

        Returns:
            flow_fields (torch.tensor): Output of network
        """
        # first the fully connected layer takes in the input, and outputs
        # 9 neurons which are reshaped into a 3x3 array
        x = self.FC_Layer(x).view(len(x), -1, 3, 3)
        # the Conv layers take in the 3x3 array and output a 163x163 array
        return self.net(x)

    @staticmethod
    def create_floris_dataset(
            size, image_size, u_range, ti_range, yaw_range,
            floris_init_path=".", curl=False):
        """
        Function to generate the dataset needed for training using FLORIS.
        The flowfield around a turbine is generated for a large range of wind
        speeds, turbulent intensities and yaw angles. The 2d array and
        correspoding init conditions are saved for training. The data is
        generated using a  Gaussian wake mode, please see:
        https://doi.org/10.1016/j.renene.2014.01.002.
        For more information about FLORIS see: https://github.com/NREL/floris.
        Function can be used to generated training, validation and test sets.

        Args:
            size (int): Size of the dataset
            image_size (int): Size of the flow field outputs that
                                        are generated, this depends on the
                                        Neural network used, should be 163.
            u_range (list): Bound of u values [u_min, u_max] used
            ti_range (list): Bound of TI values [TI_min, TI_max] used
            yaw_range (list): Bound of yaw angles [yaw_min, yaw_max] used
            floris_init_path (str, optional): Path to the FLORIS jason file.
                                              Defaults to ".".
            curl (bool, optional): If curl model should be used please set
                                   to True, see this for more information:
                                   https://doi.org/10.5194/wes-4-127-2019.
                                   Defaults to False.

        Returns:
            y (torch.tensor): Tensor of size (size, image_size, image_size)
            which includes all the generated flow fields. The flow fields
             are normalised to help training
            x (torch.tensor): Tensor of size (size, 1, 3) which includes the
            flow conditons of the correspoding flow field in the x tensor.
        """

        # sample u, ti and yawn angle from a uniform distribution
        u_list = [round(random.uniform(u_range[0], u_range[1]), 2) for
                  i in range(0, size)]
        ti_list = [round(random.uniform(ti_range[0], ti_range[1]), 2) for
                   i in range(0, size)]
        yawn_list = [round(random.uniform(yaw_range[0], yaw_range[1]), 2) for
                     i in range(0, size)]

        # initialize FLORIS model using the jason file
        if curl is False:
            floris_turbine = wfct.floris_interface.FlorisInterface(
                floris_init_path + "/FLORIS_input_gauss.json")
        else:
            floris_turbine = wfct.floris_interface.FlorisInterface(
                floris_init_path + "/FLORIS_input_curl.json")

        # initialize empty numpy array to store 2d arrays and
        # corresponding u, ti and yawn values
        y = np.zeros((size, image_size, image_size))
        x = np.zeros((size, 3))

        # create train examples
        print("generate FLORIS data")
        for _ in range(0, size):
            if _ % 500 == 0:
                print(f"{_}/{size}")
            # set wind speed, ti and yawn angle for FLORIS model
            floris_turbine.reinitialize_flow_field(
                wind_speed=u_list[_],
                turbulence_intensity=ti_list[_])
            floris_turbine.change_turbine([0], {'yaw_angle': yawn_list[_]})

            # calculate the wakefield
            floris_turbine.calculate_wake()
            # extract horizontal plane at hub height
            cut_plane = floris_turbine.get_hor_plane(
                height=90,
                x_resolution=image_size,
                y_resolution=image_size,
                x_bounds=[0, 3000],
                y_bounds=[-200, 200]).df.u.values.reshape(image_size,
                                                          image_size)
            # save the wind speed values of the plane at hub height and
            # the corresponding turbine stats
            y[_] = cut_plane
            x[_] = u_list[_], ti_list[_], yawn_list[_]

        # turn numpy array into a pytroch tensor
        x = torch.tensor(x, dtype=torch.float)
        # The wind speeds are normalised by dividing it by 12
        # i.e. every value will be between 0-1 which helps training
        y = torch.tensor(y, dtype=torch.float)/12

        return x, y

    def error(self, x_eval, y_eval, device, image_size=163, normalisation=12):
        r"""
        Calculate the average pixel wise percentage error of the model on
        a evaluation set. For error function is:
        error = 1/set_size *\sum_{n=0}^{set_size}(1/image_size**2 *
                \sum_{i=0}^{image_size**2}(100*abs(FLORIS_{n,i} - GAN_{n,i})/
                max(FLORIS_{n,i})))
        For a detailed explanation of this function please see the report in
        the ACSE9 repo.
        """
        error_list = []
        # Use model to predict the wakes for the given conditions in x
        model_predict = self.forward(x_eval.to(device))
        for n in range(0, len(x_eval)):
            # Calculate the mean error between CNNwake output and FLORIS
            # for a given flow field using the function given above
            pixel_error = np.sum(abs(
                    y_eval.detach().cpu().numpy()[n] -
                    model_predict.squeeze(1)[n].detach().cpu().numpy()) /
                    (torch.max(y_eval.detach()[n]).cpu().numpy()))
            # divide by number of pixels in array for an mean value
            pixel_error /= image_size * image_size
            error_list.append(pixel_error * 100)

        # return mean error
        return np.mean(error_list)

    def evaluate_model(self, set_size, u_range, ti_range, yaw_range,
                       image_size=163, device='cpu', normalisation=12,
                       florisjason_path="."):
        """
        Function to calculate a average pixel wise percentage error
        of the model using the error function. This functions generates
        a test set and evaluates the model on this unseen data to provide
        a test error.

        Args:
            set_size (int, optional): Nr. of samples to be used for testing.
            u_range (list): Bound of u values [u_min, u_max] used
            ti_range (list): Bound of TI values [TI_min, TI_max] used
            yaw_range (list): Bound of yaw angles [yaw_min, yaw_max] used
            image_size (int, optional): Size of the flow field.
             Defaults to 163.
            device (str): Device to store and run the neural network on,
             either cpu or cuda.
            normalisation (int, optional): The CNN output is between
             0 and 1 due to the  normalisation used, therefore it needs to
             be renormalised. Defaults to 12.
            florisjason_path (str, optional): Location of the FLORIS jason
             file. Defaults to ".".

        Returns:
           error (float): Error of model on test set
        """
        # Create a dataset to test the model on
        x_eval, y_eval = self.create_floris_dataset(
            set_size, image_size, u_range=u_range, ti_range=ti_range,
            yaw_range=yaw_range, floris_init_path=florisjason_path)

        # Use generated data set to calculate the error of CNNwake when
        # compared with the FLORIS output
        test_error = self.error(x_eval, y_eval, device,
                                image_size, normalisation=12)

        return test_error

    def epoch_training(self, criterion, optimizer, dataloader, device):
        """
        Trains the model for one epoch data provided by dataloader. The model
        will be updated after each batch and the function will return the
        train loss of the last batch

        Args:
            criterion (torch.nn.criterion): Loss function used to train model
            optimizer (torch.optim.Optimizer): Optimizer for gradient descent
            dataloader (torch.utils.DataLoader): Dataloader to store dataset
            device (str): Device on which model/data is stored, cpu or cuda

        Returns:
            training loss (float): Loss of training set defined by criterion
        """
        # For all training data in epoch
        for real_images, label in dataloader:
            # move data to device
            real_images = real_images.to(device)
            label = label.to(device)
            # images need to be in correct shape: batch_size x 1 x 1 x 3

            # compute reconstructions of flow-field using the CNN
            outputs = self.forward(label)

            # compute training reconstruction loss using the
            # loss function set
            train_loss = criterion(outputs, real_images)

            optimizer.zero_grad()  # Zero gradients of previous step
            train_loss.backward()  # compute accumulated gradients
            optimizer.step()  # Do optimizer step

        # return training loss
        return train_loss.item()

    def load_model(self, path='.', device='cpu'):
        """
        Function to load model from a pt file into this class.

        Args:
            path (str): path to saved model.
            device (torch.device): Device to load onto, cpu or cuda

        """
        # load the pretrained model
        self.load_state_dict(torch.load(path, map_location=device))

    def save_model(self, name='generator.pt'):
        """
        Function to save current model paramters so that it can
        be used again later. Needs to be saved with as .pt file

        Args:
            name (str): name of .pt file from which to load model
        """
        # Save current model parameters
        torch.save(self.state_dict(), name)
