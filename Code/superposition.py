from re import S
from neuralWake import *
from torch import cpu
from CNNWake.FCC_model import *

warnings.filterwarnings("ignore")

# Synth value
if train_net == 0:
    # Load model
    model = wakeNet().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval().to(device)
# Use CNNWake module to calculate local ti values
# Initialise network to local turbulent intensities
nr_input_values = 42  # Number of input values
nr_neurons_ti = 200  # Number of neurons in every layer
nr_neurons = 300  # Number of neurons in every layer
nr_output = 1  # Number of outputs from model
# Use CNNWake module to calculate local power and ti values
if local_ti == True:
    TI_model = FCNN(nr_input_values, nr_neurons_ti, nr_output).to(device)
    # Load trained model and set it to evaluation mode
    TI_model.load_model("CNNWake/FCNN_TI.pt", device=device)
    TI_model.eval()
if local_pw == True:
    pw_model = FCNN(nr_input_values, nr_neurons, nr_output).to(device)
    # Load trained model and set it to evaluation mode
    pw_model.load_model("CNNWake/power_model.pt", device=device)
    pw_model.eval()


def florisPw(u_stream, tis, xs, ys, yws):
    # Initialise FLORIS for initial configuraiton
    if curl == True:
        fi.floris.farm.set_wake_model("curl")
    fi.reinitialize_flow_field(wind_speed=u_stream)
    fi.reinitialize_flow_field(turbulence_intensity=tis)
    fi.reinitialize_flow_field(layout_array=[xs, ys])
    fi.calculate_wake(yaw_angles=yws)
    # Get initial FLORIS power
    floris_power_0 = fi.get_farm_power()

    return round(floris_power_0, 2)


def superposition(
    inpt1,
    inpt2,
    u_stream,
    tis,
    cp=None,
    wind_speed=None,
    farm_opt=False,
    plots=False,
    power_opt=True,
    print_times=False,
    timings=False,
    floris_gain=False,
    x0=np.zeros(1),
    single=False,
    saveas=None,
):
    """
    Calls the neural model to produce neural wakes and superimposes them on the
    computational domain in order to calculate the total farm power output in MW.

    Args:
        yws (numpy array of floats) Yaws of each turbine.
        u_stream (float) Free stream velocity.
        tis (numpy array of floats) Yaws of each turbine.
        xs (numpy array of floats) Turbine x coordinates.
        ys (numpy array of floats) Turbine y coordinates.
        cp (numpy array of floats) Cp values of turbine Cp-wind speed curve.
        wind_speed (numpy array of floats) Wind speed values of turbine Cp-wind speed curve.
        plots (boolean, optional) If True, Plots superimposed wakes.
        power_opt (boolean, optional) If True, performs one optimization step.
        print_times (boolean, optional) If True, prints timings.
        timings (boolean, optional) Returns model timings.
        floris_gain (boolean, optional) If True, calculates and returns gained power output
            with the optimised results of the DNN but using Floris for comparison.
        x0 (numpy array, optional) Defined with size > 1 for farm optimisations for storing
            initial turbine coordinates.

    Returns:
        floris_time (float) Time required for Floris computation.
        neural_time (float) Time required for a forward solution of the DNN.

        or

        -power_tot (float) Total (negative) farm power output produced by the DNN,
            based on the input turbine yaws and positions.
        floris_power_opt (float) Total farm power output produced by Floris in MW,
            based on the input turine yaws and positions.
    """

    # Local pw
    pw_ar = []
    # Scales final domain
    xscale = 0.7

    if curl == True:
        fi.floris.farm.set_wake_model("curl")

    # Select first and second argument based on the optimisiation mode.
    # Scipy's "minimise" prefers the parameter of optimisaion to be first.
    if farm_opt == True:
        layout = inpt1
        yws = inpt2
    else:
        layout = inpt2
        yws = inpt1

    # Save initial positions. x0 defined only for farm optimisation.
    if x0.size > 1:
        xs0 = x0[: int(layout.size / 2 + 0.25)]
        ys0 = x0[int(layout.size / 2 + 0.25) :]
        xs0_arg = xs0.argsort()
        xs0 = xs0[xs0_arg]
        ys0 = ys0[xs0_arg]

    # Split layout vector in x and y coordinates
    layout = np.array(layout)
    xs = layout[: int(layout.size / 2 + 0.25)]
    ys = layout[int(layout.size / 2 + 0.25) :]

    # Sort x, y and yaws based on x coordinates to superimpose
    # the turbines from left to right (downstream direction).
    xs_arg = xs.argsort()
    xs = xs[xs_arg]
    ys = ys[xs_arg]
    yws = np.array(yws)
    yws = yws[xs_arg]

    # Initialisations
    n = xs.size             # Turbine number
    clean = np.zeros(n)
    if n == 1: single = True
    hbs = 90                # Hub height
    inlet_speed = u_stream  # Speed at inlet

    # Domain dimensions
    x_domain = x_bounds[1] - x_bounds[0]
    y_domain = y_bounds[1] - y_bounds[0]

    # Hub speeds and Yaws' initialization
    hub_speeds = np.zeros(n)
    hub_speeds_power = np.zeros(n)
    hub_speeds_mean = np.zeros(n)

    # Define dx, dy
    dx = np.abs(x_domain / dimx)
    dy = np.abs(y_domain / dimy)

    # Domain dimensions
    length = np.max(np.abs(xs)) + x_domain
    domain_cols = int(length / dx + .5)
    height = 2 * np.max(np.abs(ys)) + y_domain
    domain_rows = int(height / dy + .5)

    # Domain shape initialization
    domain = np.ones((domain_rows, domain_cols)) * inlet_speed
    neural_old = np.ones((dimy, dimx)) * inlet_speed

    # Calculate the position of the first wake in the domain.
    rows1 = int(domain_rows / 2 - ys[0] / dy - dimy / 2 + .5)
    rows2 = int(domain_rows / 2 - ys[0] / dy + dimy / 2 + .5)
    cols1 = int(xs[0] / dx + .5)
    cols2 = int(xs[0] / dx + .5) + dimx

    # Start DNN timer
    t0 = time.time()
    for p in range(n):

        # Define start and finish rows of the current turbine's hub
        hub_start = int((rows2 + rows1) / 2 - D / dy / 2 + .5)
        hub_finish = int((rows2 + rows1) / 2 + D / dy / 2 + .5)
        hub_tot = hub_finish - hub_start
        if np.all(domain[hub_start:hub_finish, cols1] == u_stream):
            clean[p] = 1
        
        # Method A (mean). Calculate the mean speed on the hub.
        inlet_speed_mean = np.mean(domain[hub_start:hub_finish, cols1])
        # Method B (rings). Numerically integrate over the rotor swept area surface.
        # This gives a better approximation to the 3D domain calculations of Floris.
        inlet_speed = 0
        inlet_speed_pw = 0
        area = np.pi * D * D / 4
        for i in range(int(hub_tot / 2)):
            # Stop calculation if the profile == u_stream
            if clean[p] == 1:
                break
            # Find mean ring speed assuming symmetric flow with respect to the tower.
            mean_hub_speed = np.mean([domain[hub_start + i, cols1], domain[hub_finish - i, cols1]])
            # # Calculate total rotor area.
            # area += 2 * np.pi * int((hub_tot/2-i)*dy) * dy
            # Calculate inlet speed of current turbine based on the current state of the domain.
            inlet_speed += (mean_hub_speed * 2 * np.pi * (int(hub_tot / 2) - i) * dy * dy)
            if local_pw != True:
                # Calculate speed^3 (kinetic energy) term that will go in the calculation of power.
                area_int = 2 * np.pi * (int(hub_tot / 2) - i) * dy * dy
                inlet_speed_pw += (mean_hub_speed * mean_hub_speed * mean_hub_speed * area_int)

        # Divide speeds by total calculated area
        inlet_speed /= area
        inlet_speed_pw /= area
        inlet_speed_pw = (inlet_speed_pw) ** (1 / 3)

        # Profile == u_stream or Single wake condition
        if clean[p] == 1 or single == True:
            inlet_speed = u_stream
            inlet_speed_pw = u_stream

        # Limit the minimum speed at the minimum training speed of the DNN.
        if inlet_speed < ws_range[0]:
            inlet_speed = ws_range[0]

        # Use CNNWake module to calculate local ti values for each turbine
        ti_ar = np.ones(2)*tis
        if local_ti == True or local_pw == True:
            speeds_50m = domain[hub_start:hub_finish, cols1 - int(50 / dx + .5)]  # ***
            sss = speeds_50m.size
            ult = np.array([((speeds_50m[i - 1] + speeds_50m[i] + speeds_50m[i + 1])/3) 
                   for i in np.linspace(1, sss-2, 40, dtype=int)])
            yaw_angle = yws[p]
            turbulent_int = tis
            ult /= 15
            # The array conists of 40 wind speeds values, the yaw angle and inflow TI
            # change the two last values of the array to yaw angle and inflow TI b4 passing to NN
            ult = np.append(ult, yaw_angle / 35)
            ult = np.append(ult, turbulent_int)
        if local_ti == True and clean[p] != 1:# and curl != 1:
            ti_norm = 0.3
            ti2 = (TI_model((torch.tensor(ult).float().to(device))).detach().cpu().numpy() * ti_norm)
            if ti2 < turbulent_int * 0.7:
                ti2 = turbulent_int * 1.5
            # clip ti values to max and min trained
            ti_ar[1] = np.clip(ti2, 0.01, 0.25).item(0)
            ti_ar[0] = tis
        if local_pw == True:
            pw_norm = 4996386
            pw = (pw_model((torch.tensor(ult).float().to(device))).detach().cpu().numpy() * pw_norm)
            pw_ar.append(pw)

        # Get the DNN result
        # print(u_stream, inlet_speed, ti_ar, yws[p], hbs)
        neural = model.compareContour(
            u_stream, inlet_speed, ti_ar, yws[p], hbs, model, result_plots=False
        )
        
        # Save the inlet speed terms
        hub_speeds[p] = inlet_speed
        hub_speeds_mean[p] = inlet_speed_mean
        hub_speeds_power[p] = inlet_speed_pw

        # Apply SOS for after the first turbine is placed in the domain
        if p != 0 and p != (xs.size):
            # Apply the SOS superposition model
            def1 = np.square(1 - neural / hub_speeds[p])
            def2 = np.square(1 - neural_old / u_stream)
            neural = (1 - np.sqrt(def1 + def2)) * u_stream

        # Apply denoise filter (mainly for plotting purposes)
        if denoise > 1:
            neural[:, 1:] = ndimage.median_filter(
                neural[:, 1:], denoise
            )  # denoise filter

        # Place the DNN output inside the domain
        domain[rows1:rows2, cols1:cols2] = neural

        # Calculate the rows and columns of the next wake inside the domain
        if p != (xs.size - 1):
            p2 = p + 1
            rows1 = int(domain_rows / 2 - ys[p2] / dy - dimy / 2 + .5)
            rows2 = int(domain_rows / 2 - ys[p2] / dy + dimy / 2 + .5)
            cols1 = int(xs[p2] / dx + .5)
            cols2 = int(xs[p2] / dx + .5) + dimx
            # Store an old image of the domain to be used in the next superposition
            neural_old = np.copy(domain[rows1:rows2, cols1:cols2])

    # End DNN timer
    t1 = time.time()

    # Print DNN time
    neural_time = t1 - t0
    neural_time_rnd = round(t1 - t0, 2)
    if print_times == True:
        print("Total Neural time: ", neural_time_rnd)

    # 2 Modes: Plot contours and/or Return calculation timings.
    if plots == True or timings == True:

        # Start FLORIS timer
        t0 = time.time()

        # Initialise FLORIS
        if curl == True:
            fi.floris.farm.set_wake_model("curl")
        fi.reinitialize_flow_field(wind_speed=u_stream)
        fi.reinitialize_flow_field(turbulence_intensity=tis)
        if single != True:
            fi.reinitialize_flow_field(layout_array=[xs, -ys])

        # Get FLORIS power
        if timings == False:
            fi.calculate_wake(yaw_angles=yaw_ini)
            floris_power_0 = fi.get_farm_power()
        fi.calculate_wake(yaw_angles=yws)
        floris_power_opt = fi.get_farm_power()

        if plots == True:
            nocut=0
        else:
            nocut=1
        if nocut != 1:
            if single == True:
                cut_plane = fi.get_hor_plane(height=hbs,
                                            x_bounds=x_bounds,
                                            y_bounds=y_bounds,
                                            x_resolution=dimx,
                                            y_resolution=dimy)
            else:
                cut_plane = fi.get_hor_plane(height=hbs,
                                            x_bounds=(0, length+0.5*dx),
                                            y_bounds=(-height/2, height/2),
                                            x_resolution=domain_cols,
                                            y_resolution=domain_rows)

            u_mesh0 = cut_plane.df.u.values.reshape(
                cut_plane.resolution[1], cut_plane.resolution[0]
            )
            if single == True:
                u_mesh = np.ones((domain_rows, domain_cols)) * inlet_speed
                u_mesh[rows1:rows2, cols1:cols2] = u_mesh0
            else:
                u_mesh = u_mesh0

        # End FLORIS timer
        t1 = time.time()
        floris_time = t1 - t0
        floris_time_rnd = round(t1 - t0, 2)
        if print_times == True:
            print("Total Floris time: ", floris_time_rnd)
        if timings == True:
            return floris_time, neural_time

        # Define plot length and height
        fx = cut_plane.df.x1.values
        new_len = np.max(fx) - np.min(fx)
        new_height1 = np.min(ys) - 2 * D
        new_height2 = np.max(ys) + 2 * D
        row_start = int(domain.shape[0] / 2 - np.max(ys) / dy - 2 * D / dy + .0)
        row_finish = int(domain.shape[0] / 2 - np.min(ys) / dy + 2 * D / dy + .5)
        # Keep the FLORIS and DNN domains to be plotted
        u_mesh = u_mesh[row_start:row_finish, :]
        domain_final = domain[row_start:row_finish, :]
        # domain_final = domain

        # Keep min and max velocities of FLORIS domain
        vmin = np.min(u_mesh)
        vmax = np.max(u_mesh)


        if u_mesh.shape != domain_final.shape:
            print("Error: unequal domain shapes!")

        # Set figure properties
        fig, axs = plt.subplots(3, sharex=False)
        cmap = "coolwarm"

        # ----- FLORIS wake plots ----- #
        if contours_on == True:
            X, Y = np.meshgrid(
                np.linspace(0, new_len, u_mesh.shape[1]),
                np.linspace(new_height2, new_height1, u_mesh.shape[0]),
            )
            contours = axs[0].contour(X, Y, u_mesh, 4, alpha=0.5, linewidths=0.5, colors="white")
            axs[0].clabel(contours, inline=False, fontsize=1)

        im1 = axs[0].imshow(
            u_mesh[:, :int(xscale*u_mesh.shape[1])],
            vmin=vmin+1.25,
            vmax=vmax,
            cmap=cmap,
            extent=[0, new_len*xscale, new_height1, new_height2],
        )
        fig.colorbar(im1, ax=axs[0])
        axs[0].tick_params(axis="x", direction="in")
        axs[0].tick_params(axis="y", direction="in", length=0)

        # ----- DNN wake plots ----- #
        if contours_on == True:
            X, Y = np.meshgrid(
                np.linspace(0, new_len, domain_final.shape[1]),
                np.linspace(new_height2, new_height1, domain_final.shape[0]),
            )
            contours = axs[1].contour(X, Y, domain_final, 1, colors="white")
            axs[1].clabel(contours, inline=True, fontsize=8)

        im2 = axs[1].imshow(
            domain_final[:, :int(xscale*domain_final.shape[1])],
            vmin=vmin+1.25,
            vmax=vmax,
            cmap=cmap,
            extent=[0, new_len*xscale, new_height1, new_height2],
        )
        fig.colorbar(im2, ax=axs[1])
        axs[1].tick_params(axis="x", direction="in")
        axs[1].tick_params(axis="y", direction="in", length=0)

        # ----- ERROR (%) plots ----- #
        max_val = np.max(u_mesh)
        im3 = axs[2].imshow(
            (np.abs(u_mesh - domain_final) / max_val * 100)[:, :int(xscale*domain_final.shape[1])],
            cmap=cmap,
            extent=[0, new_len*xscale, new_height1, new_height2],
            vmax=20,
        )
        axs[2].tick_params(axis="x", direction="in")
        axs[2].tick_params(axis="y", direction="in", length=0)
        plt.colorbar(im3, ax=axs[2])

        if saveas != None:
            fig.savefig("figures/"+str(saveas), dpi=1200)
        else:
            plt.show()

        absdifsum = np.sum(np.abs(u_mesh - domain_final))
        error = round(1 / (dimx * dimy) * absdifsum / max_val * 100, 2)
        print("Abs mean error (%): ", error)

        # ----- Y-Transect plots ----- #
        mindx = np.min(xs)/dx+0.5
        mindx = int(mindx)
        tlist = mindx + np.array([3*D/dx, 6.5*D/dx, 10*D/dx]).astype(int)
        transects = tlist.size  # defines the number of transects
        fig, axs = plt.subplots(1, transects, sharey=False)
        cnt = 0
        for indx in tlist:

            yy1 = u_mesh[:, indx]  # FLORIS transect
            yy2 = domain_final[:, indx]  # CNN transect
            axs[cnt].plot(
                np.flip(yy1, axis=0),
                np.arange(u_mesh.shape[0]),
                color="navy",
                linestyle="--",
            )
            axs[cnt].plot(
                np.flip(yy2, axis=0), np.arange(u_mesh.shape[0]), color="crimson"
            )
            axs[cnt].title.set_text(str(int(indx * dx)))
            cnt += 1

        if saveas != None:
            fig.savefig("figures/"+str(saveas)+"yt", dpi=1200)
        else:
            plt.show()

    if power_opt == True:

        # Calculation of total farm power

        if local_pw == True:
            power_tot = pw_ar
        else:
            rho = 1.225  # air density
            hub_speeds_old = np.copy(hub_speeds_power)

            # Interpolate cp values
            cp_interp = np.interp(hub_speeds_old, wind_speed, cp)

            # Multiply by cos(theta) term
            # Default exponent of cos term is 1.0.
            # An exponent of .78 was found to perform best.
            cp_interp *= np.cos(np.pi / 180 * (-yws)) ** (1.3)

            # Calculate powers using the kinetic energy term
            power_tot = 0.5 * rho * cp_interp * hub_speeds_power**3 * area
        
        # Sum of all turbine power outputs
        power_tot = np.sum(power_tot)
        
        if floris_gain == True:
            # Calculate power gain as provided by FLORIS (for final assessment of optimisation).

            # Initialise FLORIS for initial configuraiton
            if curl == True:
                fi.floris.farm.set_wake_model("curl")
            fi.reinitialize_flow_field(wind_speed=u_stream)
            fi.reinitialize_flow_field(turbulence_intensity=tis)
            if x0.size > 1:
                fi.reinitialize_flow_field(layout_array=[xs0, ys0])
            else:
                fi.reinitialize_flow_field(layout_array=[xs, ys])
            
            fi.calculate_wake(yaw_angles=yaw_ini)
            # Get initial FLORIS power
            floris_power_0 = fi.get_farm_power()
            floris_power_opt = florisPw(u_stream, tis, xs, ys, yws)
            
            floris_power_gain = round(
                (floris_power_opt - floris_power_0) / floris_power_0 * 100, 2
            )

            if plots == True:
                print("----------------FLORIS for Neural--------------------")
                print("Floris Initial Power", round(floris_power_0 / 1e6, 2), "MW")
                print("Floris Optimal power", round(floris_power_opt / 1e6, 2), "MW")
                print("Floris Power Gain (%)", floris_power_gain)
                print("-----------------------------------------------------")
            return -power_tot, floris_power_opt/1e6, floris_power_0/1e6

        else:
            # Calculate power gain as provided by the DNN (used in optimisation steps).
            return -power_tot
