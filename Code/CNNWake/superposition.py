import torch
from torch.backends import cudnn
import matplotlib.pyplot as plt
import numpy as np
import time
import floris.tools as wfct

__author__ = "Jens Bauer"
__copyright__ = "Copyright 2021, CNNwake"
__credits__ = ["Jens Bauer"]
__license__ = "MIT"
__version__ = "1.0"
__email__ = "jens.bauer20@imperial.ac.uk"
__status__ = "Development"


def super_position(farm_array, turbine_array, turbine_postion,
                   hub_speed, wind_velocity, sp_model="SOS"):
    """
    Generate super-position of a turbine wind field and a farm wind field.
    The turbine wind field is superimposed onto the wind farm flow field using
    different super-postion models. The recommended model is the root sum of
    squares (SOS), more information about super-postion can be found in this
    paper: https://doi.org/10.1088/1742-6596/749/1/012003

    Args:
        farm_array (numpy array): 2d wind field of whole wind farm
        turbine_array (numpy array): 2d wind field around wind turbine
        turbine_postion (numpy array): x and y cell number of wind turbine
            in the global array [x_cell, y_cell]
        hub_speed (float): u velocity at turbine hub in m/s
        wind_U_turbine (float): wind speed at turbine hub
        wind_velocity (float): free stream wind speed
        sp_model (str, optional): Select model to be used for the
         super-positioning  Defaults to "SOS".

    Returns:
        numpy array: 2d wind field of whole wind farm with flow
                     field around turbine superimposed
    """
    # normalize wind by the free stream and hub height speed
    turbine_u = turbine_array/hub_speed
    farm_u = farm_array/wind_velocity

    # Define the start and end coordinates of the turbine wake
    # in the global wind park array
    x_start = turbine_postion[0]
    x_end = turbine_postion[0]+turbine_u.shape[0]
    y_start = turbine_postion[1]
    y_end = turbine_postion[1]+turbine_u.shape[1]

    if sp_model == "SOS":
        # For SOS model with one turbine, the equation is:
        # u = 1 - sqrt((1 - u_1)^2 + (1 - u_2)^2)
        sos1 = np.square(1 - turbine_u)
        sos2 = np.square(1 - farm_u)

        # place the SOS superpostion in the correct location of the farm array
        farm_array[y_start:y_end, x_start:x_end] = (1 - np.sqrt(
            sos1 + sos2[y_start:y_end, x_start:x_end])) * wind_velocity

        # farm_array now includes the velocity field of the turbine
        return farm_array

    elif sp_model == "linear":
        # For SOS model with one turbine, the equation is:
        # u = 1 - ((1 - u_1) + (1 - u_2))
        sos1 = 1 - turbine_u
        sos2 = 1 - farm_u

        # place the linear superpostion in the correct
        # location of the farm array
        farm_array[
            turbine_postion[1]:turbine_postion[1]+sos1.shape[1],
            turbine_postion[0]:turbine_postion[0]+sos1.shape[0]] = \
            (1 - (sos1 + sos2[
                turbine_postion[1]:turbine_postion[1]+sos1.shape[1],
                turbine_postion[0]:turbine_postion[0]+sos1.shape[0]]))\
            * wind_velocity

        # farm_array now includes the velocity field of the turbine
        return farm_array

    elif sp_model == "largest_deficit":
        # u = min(u_1, u_2)
        # place the SOS super postion in the correct location of the farm array
        farm_array[
            turbine_postion[1]:turbine_postion[1]+turbine_u.shape[1],
            turbine_postion[0]:turbine_postion[0]+turbine_u.shape[0]]\
            = np.minimum(turbine_u,
                         farm_u[y_start:y_end,
                                x_start:x_end]) * wind_velocity

        # farm_array now includes the velocity field of the turbine
        return farm_array

    else:
        # other models to be added
        raise Exception('No super position model selected, please'
                        ' either select: SOS, linear or largest_deficit')


def CNNWake_farm_power(
        yawn_angles, x_position, y_position, wind_velocity, turbulent_int,
        CNN_generator, Power_model, TI_model, device,
        ti_normalisation=0.30000001, power_normalisation=4834506):

    """
    Calculates the power output of the wind farm using the NN.
    The generated power is returned as negative number for the minimization.
    The individual wakes of the turbines are calculated using the CNN and
    superimposed onto the wind farm flow field using a super-position model.
    The energy produced by the turbines are calculated using another fully
    connected network from the flow data just upstream the turbine.
    Please ensure that the x position are in ascending order and every
    turbine is placed at least 300 above 0 in the direction. This is done
    to ensure that no wake is lost at the edge of the domain.

    Args:
        yawn_angles (list): 1d array of the yaw angle of every wind turbine
            in degree, from -30° to 30°
        x_position (list): 1d array of the x postions of the wind
            turbines in meters.
        y_position (list): 1d array of the y postions of the wind
            turbines in meters.
        wind_velocity (float): Free stream wind velocity in m/s,
            from 3 m/s to 12 m/s
        turbulent_int (float): Turbulent intensity in percent,
            from 1.5% to 25%
        CNN_generator (Generator): CNN to predict the wake of a single
             turbine, ensure it is trained and set to validation mode
        Power_model (Generator): FCNN to predict the power generated
             by a turbine, ensure it is trained and set to validation mode
        TI_model (Generator): FCNN to predict the local TI of a
             turbine, ensure it is trained and set to validation mode
        device (torch.device): Device to store and run the neural network
            on, either cpu or cuda
        ti_normalisation (float): Normalisation of the TI training set
        power_normalisation (float): Normalisation of the power training set

    Returns:
        power float: negative power output
    """
    # Define the x and y length of a single cell in the array
    # This is set by the standard value used in FLORIS wakes
    dx = 18.4049079755
    dy = 2.45398773006
    # Set the maximum length of the array to be 3000m and 400m
    # more than the maximum x and y position of the wind park
    # If a larger physical domain was used change adapt the values
    x_max = np.max(x_position) + 3000
    y_max = np.max(y_position) + 300
    # Number of cells in x and y needed to create a 2d array of
    # that is x_max x y_max using dx, dy values
    Nx = int(x_max/dx)
    Ny = int(y_max/dy)
    # Initialise a 2d array of the wind park with the
    # inlet wind speed
    farm_array = np.ones((Ny, Nx)) * wind_velocity

    # round yaw angle
    yawn_angles = np.round(yawn_angles, 2)

    # Initialise array to store power and TI for every turbine
    power_CNN = []
    ti_CNN = []

    with torch.no_grad():  # Ensure no gradients are calculated
        # For every wind turbine
        for i in range(len(x_position)):
            # determine the x and y cells that the turbine center is at
            turbine_cell = [int((x_position[i])/dx),
                            int((y_position[i] - 200)/dy)]
            # extract wind speeds along the rotor, 60 meters upstream
            u_upstream_hub = farm_array[
                turbine_cell[1] + 45: turbine_cell[1] + 110, turbine_cell[0] - 3]
            # Do an running average, this is done because CNNwake has slight
            # variations in the u predictions, also normalise the u values
            u_list_hub = [
                    ((u_upstream_hub[i-1] + u_upstream_hub[i] +
                      u_upstream_hub[i+1])/3)/12 for i in np.linspace(
                        5, len(u_upstream_hub)-5, 40, dtype=int)]
            # append yaw angle and normalised it, also append ti
            u_list_hub = np.append(u_list_hub, yawn_angles[i]/30)
            u_list_hub = np.append(u_list_hub, turbulent_int)

            # The local TI does not change from inlet TI if the turbine
            # is not covered by a wake, therefore check if if all values
            # in u_list_hub are the same -> means no wake coverage
            # Local TI also depends on yaw, if yaw is less than 12° and
            # turbine is not in wake -> use inlet TI for local TI
            if np.allclose(
                    u_list_hub[0], u_list_hub[0:-3], rtol=1e-02, atol=1e-02)\
                    and abs(u_list_hub[-2]) < 0.4:
                ti = turbulent_int
            # If turbine is in wake or yaw angle is larger use FCNN to find
            # local TI
            else:
                # Use FCNN forward pass to predict TI
                ti = TI_model((torch.tensor(u_list_hub).float().to(device))).detach().cpu().numpy() * ti_normalisation
                # regulate TI to ensure it is not to different from free stream
                if ti < turbulent_int*0.7:
                    ti = turbulent_int * 1.5
                # clip ti values to max and min trained
                ti = np.clip(ti, 0.015, 0.25).item(0)
            ti_CNN.append(ti)  # Save ti value

            # Replace global/inlet TI in u_list with local TI
            u_list_hub[-1] = ti
            # Use FCNN to predcit power generated by turbine
            turbine_energy = Power_model(torch.tensor(u_list_hub).float().to(device)).detach().cpu().numpy() * power_normalisation
            power_CNN.append(turbine_energy)  # Save power

            # Find the mean wind speed upstream the turbine
            hub_speed = np.round(np.mean(u_upstream_hub), 2)
            # Create Array of array to pass it to CNN
            turbine_condition = [[hub_speed, ti, yawn_angles[i]]]

            # Use CNN to calculate wake of individual trubine
            turbine_field = CNN_generator(torch.tensor(turbine_condition).float().to(device))
            # Since CNN output is normalised,
            # mutiply by 12 and create a numpy array
            turbine_field = turbine_field[0][0].detach().cpu().numpy() * 12
            # Place wake of indivual turbine in the farm_array
            farm_array = super_position(
                farm_array, turbine_field, turbine_cell,
                hub_speed, wind_velocity, sp_model="SOS")

    # Return the value negative of power generated
    return -sum(power_CNN).item(0)


def FLORIS_farm_power(
        yawn_angles, x_position, y_position, wind_velocity,
        turbulent_int, floris_park):
    """
    Function to generate the power output of a wind farm defined by
    the x, y and yaw angles of every turbine in the farm.
    The function will only use FLORIS to calcaute the power and
    returnes the power as a negtive value which is needed for
    the minimisation.

    Args:
        yawn_angles (list): Yaw angle of every turbine in the wind park
        x_position (list): All x locations of the turbines
        y_position (list): All y locations of the turbines
        wind_velocity (float): Inlet wind speed
        turbulent_int (float): Inlet turbulent intensity
        floris_park (floris.tools.FlorisInterface): Floris interface
            loads in data from jason file

    Returns:
        power (float): negative power generated by wind park
    """
    # Round yaw angle input
    yawn_angles = np.round(yawn_angles, 2)

    # Set the x and y postions of the wind turbines
    floris_park.reinitialize_flow_field(layout_array=[x_position,
                                                      np.array(y_position)])
    # Set the yaw angle of every turbine
    for _ in range(0, len(x_position)):
        floris_park.change_turbine([_],
                                   {'yaw_angle': yawn_angles[_],
                                    "blade_pitch": 0.0})
    # Set inlet wind speed and TI
    floris_park.reinitialize_flow_field(wind_speed=wind_velocity,
                                        turbulence_intensity=turbulent_int)
    # Calculate wind field
    floris_park.calculate_wake()
    # Calculate power generated by every turbine
    power = floris_park.get_turbine_power()

    # Return the sum of all power per turbine but as negative
    # value for the optimisation
    return -sum(power)
