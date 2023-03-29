from scipy.optimize import minimize
import numpy as np
import torch
import time
import floris.tools as wfct
from .superposition import CNNWake_farm_power, FLORIS_farm_power
from .CNN_model import Generator
from .FCC_model import FCNN

__author__ = "Jens Bauer"
__copyright__ = "Copyright 2021, CNNwake"
__credits__ = ["Jens Bauer"]
__license__ = "MIT"
__version__ = "1.0"
__email__ = "jens.bauer20@imperial.ac.uk"
__status__ = "Development"


def CNNwake_wake_steering(x_position, y_position, initial_yaw, wind_velocity,
                          turbulent_int, CNN_generator, Power_model, TI_model,
                          device, bounds, tolerance):
    """
        Function will optimise the yaw angle of a specific wind farm for
        a given inlet wind speed and TI using CNNwake's wind farm function.
        Please ensure that the x position are in ascending order and every
        turbine is placed at least 300 above 0 in the y direction. This is done
        to ensure that no wake is lost at the edge of the domain.

        Args:
            x_position (list or numpy array): 1d array of the x postions of
             the wind turbines in m.
            y_position (list or numpy array): 1d array of the y postions of
             the wind turbines in m.
            initial_yaw (list or numpy array): 1d array of inital yaw angle
             of every wind turbine in degree, set to 0
            wind_velocity (float): Free stream wind velocity in m/s,
             ensure NNa are trained on this wind speed
            turbulent_int (float): Turbulent intensity in percent ,
             ensure NNs are trained on this TI
            CNN_generator (Generator): CNN to predict the wake of a single
             turbine, ensure it is trained and set to validation mode
            Power_model (Generator): FCNN to predict the power generated
             by a turbine, ensure it is trained and set to validation mode
            TI_model (Generator): FCNN to predict the local TI of a
             turbine, ensure it is trained and set to validation mode
            device (torch.device): Device to store and run the neural network
             on, either cpu or cuda
            bounds (list): Yaw angle bounds for optimisation [min_yaw, max_yaw]
            tolerance (float): Relative solver tolerance

        Returns:
            opt_yaw.x (np.array): Optimal yaw angle
            opt_yaw.func (float): Optimal power output
            time_taken (float): Time taken for optimisation
    """

    # Set all NNs to evaluation mode
    CNN_generator.eval()
    Power_model.eval()
    TI_model.eval()

    # Run a few check to ensure that optimisation will work
    # Check if there are same number of turbines defined in
    # x, y and yaw anf´gle arrays
    assert len(x_position) == len(y_position)
    assert len(y_position) == len(initial_yaw)
    # check if x_list in ascending order, if this assert fails
    # ensure that x goes from smallest to largest
    if len(x_position) > 1:
        assert np.any(np.diff(np.array(x_position)) > 0)
    # Check if all the NNs work as expected
    assert CNN_generator(torch.tensor([[
        4, 0.1, 20]]).float().to(device)).size() == \
        torch.Size([1, 1, 163, 163])
    assert TI_model(torch.tensor([
        i for i in range(0, 42)]).float().to(device)).size() == \
        torch.Size([1])
    assert Power_model(torch.tensor([
        i for i in range(0, 42)]).float().to(device)).size() == \
        torch.Size([1])

    # create a list of tuples of bounds for the optimizer
    bounds_list = [(bounds[0], bounds[1]) for _ in range(0, len(x_position))]

    init_t = time.time()  # start timer
    # Using scipy.optimize function to find the optimal yaw setting by calling
    # CNNWake_farm_power many times with different yaw angles. Ensure that all
    # arguments are given in the correct order
    opt_yaw = minimize(
        CNNWake_farm_power, initial_yaw,
        args=(x_position, y_position, wind_velocity, turbulent_int,
              CNN_generator, Power_model, TI_model, device), method='SLSQP',
        bounds=bounds_list, options={'ftol': tolerance, 'eps': 0.1,
                                     'disp': False})

    # find time taken for optimisation
    time_taken = time.time() - init_t

    return np.round(opt_yaw.x, 2), abs(opt_yaw.fun), time_taken


def FLORIS_wake_steering(x_position, y_position, initial_yaw, wind_velocity,
                         turbulent_int, bounds, tolerance, floris_path='./'):
    """
        Function will optimise the yaw angle of a specific wind farm for
        a given inlet wind speed and TI using FLORIS.
        Please ensure that the x position are in ascending order and every
        turbine is placed at least 300 above 0 in the direction. This is done
        to ensure that no wake is lost at the edge of the domain.

            Args:
                x_position (list or numpy array): 1d array of the x postions of
                 the wind turbines in m.
                y_position (list or numpy array): 1d array of the y postions of
                 the wind turbines in m.
                initial_yaw (list or numpy array): 1d array of inital yaw angle
                 of every wind turbine in degree, set to 0
                wind_velocity (float): Free stream wind velocity in m/s,
                 ensure NNa are trained on this wind speed
                turbulent_int (float): Turbulent intensity in percent ,
                 ensure NNs are trained on this TI
                bounds (list): Yaw angle bounds for optimisation [min, max]
                tolerance (float): Relative solver tolerance
                floris_path (str): Path to FLORIS jason file

            Returns:
                floris_opti.x (np.array): Optimal yaw angle
                floris_opti.func (float): Optimal power output
                time_taken (float): Time taken for optimisation
        """

    # Check if there are same number of turbines defined in
    # x, y and yaw anf´gle arrays
    assert len(x_position) == len(y_position)
    assert len(y_position) == len(initial_yaw)

    # create a list of tuples of bounds for the optimizer
    bounds = [(bounds[0], bounds[1]) for _ in range(0, len(x_position))]

    # This variable is used to sum up all the power generated by turbines
    floris_park = 0
    # Check if path to FLORIS jason file is correct by testing if it can
    # open it
    try:
        floris_park = wfct.floris_interface.FlorisInterface(
            floris_path + "FLORIS_input_gauss.json")
    except FileNotFoundError:
        print('No FLORIS_input_gauss.jason file found at this lcoation, '
              'please specfiy the path to this file')

    init_t = time.time()  # Start timer
    # Using scipy.optimize function to find the optimal yaw setting by calling
    # FLORIS_farm_power many times with different yaw angles. Ensure that all
    # arguments are given in the correct order
    floris_opti = minimize(
        FLORIS_farm_power, initial_yaw,
        args=(x_position, y_position, wind_velocity,
              turbulent_int, floris_park),
        method='SLSQP', bounds=bounds,
        options={'ftol': tolerance, 'eps': 0.1,
                 'disp': False})

    time_taken = time.time() - init_t

    return np.round(floris_opti.x, 2), abs(floris_opti.fun), time_taken


if __name__ == '__main__':
    # To run individual CNNWake files, the imports are not allowed to be
    # relative. Instead of: from .CNN_model import Generator
    # it needs to be: from CNN_model import Generator for all CNNWake imports

    # select device to run model on
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and set up all NNs
    CNN_generator = Generator(3, 30).to(device)
    CNN_generator.load_model('./trained_models/CNN_FLOW.pt', device=device)
    Power_model = FCNN(42, 300, 1).to(device)
    Power_model.load_state_dict(torch.load('./trained_models/FCNN_POWER.pt',
                                           map_location=device))
    TI_model = FCNN(42, 300, 1).to(device)
    TI_model.load_state_dict(torch.load('./trained_models/FCNN_TI.pt',
                                        map_location=device))

    # Use optimisation to find best yaw angle
    yaw1, power1, timing1 = CNNwake_wake_steering(
        [100, 100, 1000, 1000],
        [300, 800, 300, 800],
        [0, 0, 0, 0], 10.6, 0.09, CNN_generator, Power_model, TI_model,
        device, [-30, 30], 1e-07)

    print(f"CNNwake optimized yaw abgle: {yaw1}")

    # Find FLORIS best yaw abgle
    yaw, power, timing = FLORIS_wake_steering(
        [100, 100, 1000, 1000],
        [300, 800, 300, 800],
        [0, 0, 0, 0], 10.6, 0.09, [-30, 30], 1e-07)
    print(f"FLORIS optimized yaw abgle: {yaw}")

