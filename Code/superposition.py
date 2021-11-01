from neuralWake import *
warnings.filterwarnings("ignore")

if synth == 0:
    # Load model
    model = torch.load(weights_path)
    model.eval()

def superposition(inpt1,
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
                  single=False):
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
        xs0 = x0[:int(layout.size/2 + 0.25)]
        ys0 = x0[int(layout.size/2 + 0.25):]
        xs0_arg = xs0.argsort()
        xs0 = xs0[xs0_arg]
        ys0 = ys0[xs0_arg]

    # Split layout vector in x and y coordinates
    layout = np.array(layout)
    xs = layout[:int(layout.size/2 + 0.25)]
    ys = layout[int(layout.size/2 + 0.25):]

    # Sort x, y and yaws based on x coordinates to superimpose
    # the turbines from left to right (downstream direction).
    xs_arg = xs.argsort()
    xs = xs[xs_arg]
    ys = ys[xs_arg]
    yws = np.array(yws)
    yws = yws[xs_arg]

    # Initialisations
    hbs = 90  # Hub height
    inlet_speed = u_stream  # Speed at inlet

    # Domain dimensions
    y_domain = y_bounds[1] - y_bounds[0]
    x_domain = x_bounds[1] - x_bounds[0]

    # Hub speeds and Yaws' initialization
    hub_speeds = np.zeros(xs.size)
    hub_speeds_power = np.zeros(xs.size)
    hub_speeds_mean = np.zeros(xs.size)
    # print('YWS', yws)

    dx = dimx / x_domain
    dy = dimy / y_domain

    length = x_domain + np.max(np.abs(xs))
    domain_cols = int(length * dimx / x_domain + 0.5)

    height = y_domain + 2*np.abs(np.max(np.abs(ys)))
    domain_rows = int(height * dimy / y_domain + 0.5)

    dx = domain_cols / length
    dy = domain_rows / height

    # Domain shape initialization
    domain = np.ones((domain_rows, domain_cols)) * inlet_speed

    # Calculate the position of the first wake in the domain.
    p = 0
    rows1 = int(domain_rows/2 - dy*ys[p] - dimy/2 + 0.5)
    rows2 = int(domain_rows/2 - dy*ys[p] + dimy/2 + 0.5)
    cols1 = int(dx*xs[p] + 0.5)
    cols2 = int(dx*xs[p] + 0.5) + dimx

    # Start DNN timer
    neural_old = np.ones((dimx, dimy)) * inlet_speed
    t0 = time.time()
    for p in range(xs.size):

        # Define start and finish rows of the current turbine's hub
        hub_start = int((rows2+rows1)/2 - dy*D/2 - 0.5)
        hub_finish = int((rows2+rows1)/2 + dy*D/2 + 0.5)
        hub_tot = hub_finish - hub_start

        # Method A (mean). Calculate the mean speed on the hub.
        inlet_speed_mean = np.mean(domain[hub_start:hub_finish, cols1])

        # Method B (rings). Numerically integrate over the rotor swept area surface.
        # This gives a better approximation to the 3D domain calculations of Floris.
        area =  np.pi*D*D/4
        # area = 0
        inlet_speed = 0
        inlet_speed_power = 0

        hub_tot -= 1
        for ii in range(int(hub_tot/2)):

            # Find mean ring speed assuming symmetric flow with respect to the tower.
            mean_hub_speed = np.mean([domain[hub_start+ii, cols1], domain[hub_finish-ii, cols1]])

            # # Calculate total rotor area.
            # area += 2 * np.pi * int((hub_tot/2-ii)/dy) * 1/dy

            # Calculate inlet speed of current turbine based on the current state of the domain.
            inlet_speed += mean_hub_speed * 2 * np.pi * (int(hub_tot/2)-ii)/dy * 1/dy

            # Calculate speed^3 (kinetic energy) term that will go in the calculation of power.
            area_int = 2 * np.pi * (int(hub_tot/2)-ii)/dy * 1/dy
            inlet_speed_power += mean_hub_speed * mean_hub_speed * mean_hub_speed * area_int

        # Divide speeds by total calculated area
        inlet_speed /= area
        inlet_speed_power /= area
        inlet_speed_power = (inlet_speed_power)**(1/3)

        # Limit the minimum speed at the minimum training speed of the DNN.
        if inlet_speed < ws_range[0]:
            inlet_speed = ws_range[0]

        # Single wake condition
        if single == True:
            inlet_speed_power = u_stream

        # if p == 0 or p == 1:
        #     inlet_speed_power = inlet_speed

        # Get the DNN result
        neural = model.compareContour(u_stream, ref_point, inlet_speed,
                                      tis, -yws[p], hbs, model)

        # Save the inlet speed terms
        hub_speeds[p] = inlet_speed
        hub_speeds_mean[p] = inlet_speed_mean
        hub_speeds_power[p] = inlet_speed_power
        
        # Apply SOS for after the first turbine is placed in the domain
        # if p != 0 and p != (xs.size):
        if p != (xs.size):

            # Filter out wakes
            neural[neural == u_stream] = hub_speeds[p]
            neural_old[neural_old == u_stream] = u_stream

            # Apply the SOS superposition model
            def1 = np.square( 1 - neural/hub_speeds[p] )
            def2 = np.square( 1 - neural_old/u_stream )
            neural = ( 1 - np.sqrt(def1 + def2) ) * u_stream

        # Apply denoise filter (mainly for plotting purposes)
        if denoise > 1:
            neural[:, 1:] = ndimage.median_filter(neural[:, 1:], denoise)  # denoise filter

        # Place the DNN output inside the domain
        domain[rows1:rows2, cols1:cols2] = neural

        # Calculate the rows and columns of the next wake inside the domain
        if p != (xs.size - 1):
            p2 = p + 1
            rows1 = int(domain_rows/2 - dy*ys[p2] - dimy/2 + 0.5)
            rows2 = int(domain_rows/2 - dy*ys[p2] + dimy/2 + 0.5)
            cols1 = int(dx*xs[p2] + 0.5)
            cols2 = int(dx*xs[p2] + 0.5) + dimx

            # Store an old image of the domain to be used in the next superposition
            neural_old = np.copy(domain[rows1:rows2, cols1:cols2])


    # End DNN timer
    t1 = time.time()

    # Print DNN time
    neural_time = t1 - t0
    neural_time_rnd = round(t1 - t0, 2)
    if print_times == True:
        print('Total Neural time: ', neural_time_rnd)


    # 2 Modes: Plot contours and/or Return calculation timings.
    if plots == True or timings == True:

        # Start FLORIS timer
        t0 = time.time()

        # Initialise FLORIS
        # fi.floris.farm.set_wake_model('curl')
        fi.reinitialize_flow_field(wind_speed = u_stream)
        fi.reinitialize_flow_field(turbulence_intensity = tis)
        fi.reinitialize_flow_field(layout_array=[xs, ys])

        if timings == False:
            fi.calculate_wake(yaw_angles=yaw_ini)
            floris_power_0 = fi.get_farm_power()

        fi.calculate_wake(yaw_angles=yws)
        floris_power_opt = fi.get_farm_power()

        cut_plane = fi.get_hor_plane(height=hbs,
                                     x_bounds=(0, length),
                                     y_bounds=(int(-height/2-0.5), int(height/2+0.5)),
                                     x_resolution=domain_cols,
                                     y_resolution=domain_rows)

        u_mesh = cut_plane.df.u.values.reshape(cut_plane.resolution[1],
                                               cut_plane.resolution[0])

        # End FLORIS timer
        t1 = time.time()

        # Print FLORIS time
        floris_time = t1 - t0
        floris_time_rnd = round(t1 - t0, 2)
        if print_times == True:
            print('Total Floris time: ', floris_time_rnd)

        if timings == True:
            # if timings == True return the calculation timings.

            return floris_time, neural_time
        
        # Define plot length and height

        row_start = int(domain.shape[0]/2 - np.max(ys)*dy - 2*D*dy+1)
        row_finish = int(domain.shape[0]/2 - np.min(ys)*dy + 2*D*dy)

        new_height1 = np.min(ys)-2*D-0.5
        new_height2 = np.max(ys)+2*D+0.5

        if xs.size == 1:
            new_len = length
        else:
            new_len = length - int(x_domain/2+0.5)

        col_start = 0
        col_finish = int(domain.shape[1] * new_len / length + 0.5)

        # Flip u_mesh
        u_mesh = np.flipud(u_mesh)
        
        # Keep min and max velocities of FLORIS domain
        vmin = np.min(u_mesh)
        vmax = np.max(u_mesh)

        # Keep the FLORIS and DNN domains to be plotted
        domain_final = domain[row_start:row_finish, col_start:col_finish]
        u_mesh = u_mesh[row_start:row_finish, col_start:col_finish]
        # Set figure properties
        fig, axs = plt.subplots(3, sharex=False)

        cmap = 'coolwarm'
        fontProperties = {'family':'serif',
            'weight' : 'normal', 'size' : 11}

        contourss = False  # if True, plots contours on top of wakes.

        # ----- FLORIS wake plots ----- #
        if contourss == True:
            X, Y = np.meshgrid(np.linspace(0, new_len, u_mesh.shape[1]), 
                               np.linspace(new_height1, new_height2, u_mesh.shape[0]))
            contours = axs[0].contour(X, Y, u_mesh, 1, colors='white')
            axs[0].clabel(contours, inline=True, fontsize=8)

        im1 = axs[0].imshow(u_mesh, vmin=vmin, vmax=vmax, cmap=cmap,
                            extent=[0, new_len, new_height1, new_height2])
        axs[0].set_yticklabels(np.flipud(axs[0].get_yticks().astype(int)), fontProperties)
        axs[0].set_xticklabels(axs[0].get_xticks().astype(int), fontProperties)
        fig.colorbar(im1, ax = axs[0])


        # ----- DNN wake plots ----- #
        if contourss == True:
            X, Y = np.meshgrid(np.linspace(0, new_len, domain_final.shape[1]),
                               np.linspace(new_height2, new_height1, domain_final.shape[0]))
            contours = axs[1].contour(X, Y, domain_final, 1, colors='white')
            axs[1].clabel(contours, inline=True, fontsize=8)

        im2 = axs[1].imshow(domain_final, vmin=vmin, vmax=vmax, cmap=cmap,
                            extent=[0, new_len, new_height1, new_height2])
        axs[1].set_xticklabels(axs[1].get_xticks().astype(int), fontProperties)
        axs[1].set_yticklabels(axs[1].get_yticks().astype(int), fontProperties)
        fig.colorbar(im2, ax = axs[1])


        # ----- ERROR (%) plots ----- #
        max_val = np.max(u_mesh)
        im3 = axs[2].imshow((np.abs(u_mesh - domain_final)/max_val*100), cmap=cmap,
                            extent=[0, new_len, new_height1, new_height2], vmax=20)
        plt.colorbar(im3, ax = axs[2])
        plt.show()

        absdifsum = np.sum(np.abs(u_mesh - domain_final))
        error = round(1/(dimx*dimy) * absdifsum/max_val * 100, 2)
        print('Abs mean error (%): ', error)
        plt.show()


        # ----- Y-Transect plots ----- #
        fig, axs = plt.subplots(1, 3, sharey=False)

        cnt = 0
        transects = 3  # defines the number of transects
        step = int(u_mesh.shape[1]/(transects+2))  # step between the downstream transects
        for indx in range(step, u_mesh.shape[1]-2*step+1, step):

            yy1 = u_mesh[:, indx]  # FLORIS transect
            yy2 = domain_final[:, indx]  # CNN transect

            axs[cnt].plot(np.flip(yy1, axis=0), np.arange(u_mesh.shape[0]), 
                          color='navy', linestyle='--')
            axs[cnt].plot(np.flip(yy2, axis=0), np.arange(u_mesh.shape[0]),
                          color='crimson')
            axs[cnt].set_xticklabels(np.flipud(axs[cnt].get_xticks().astype(int)), fontProperties)
            axs[cnt].set_yticklabels([], fontProperties)
            axs[cnt].title.set_text(str(int(indx/dx)))
            # axs[cnt].set_yticklabels((np.ones(2)*indx/dx).astype(int), fontProperties)
            # axs[cnt].set_xticklabels(np.arange((u_mesh.shape[0])/dy).astype(int), fontProperties)

            axs[cnt].tick_params(axis='x', direction='in')
            axs[cnt].tick_params(axis='y', direction='in', length=0)
            cnt += 1

        plt.show()

    if power_opt == True:

        # Calculation of total farm power

        rho = 1.225  # air density
        hub_speeds_old = np.copy(hub_speeds)
        # hub_speeds_old = np.copy(hub_speeds_mean)
        # hub_speeds_old = np.copy(hub_speeds_power)

        # Interpolate cp values
        cp_interp = np.interp(hub_speeds_old, wind_speed, cp)

        # Multiply by cos(theta) term
        cp_interp *= np.cos(np.pi/180*(-yws))**(1.0)  # ref 2.0

        # Calculate powers using the kinetic energy term
        power_tot = 0.5*rho*cp_interp*hub_speeds**3*area

        # Sum of all turbine power outputs
        power_tot = np.sum(power_tot)
        
        if floris_gain == True:
            # Calculate power gain as provided by FLORIS
            # (for final assessment of optimisation).

            # Initialise FLORIS for initial configuraiton
            # fi.floris.farm.set_wake_model('curl')
            fi.reinitialize_flow_field(wind_speed = u_stream)
            fi.reinitialize_flow_field(turbulence_intensity = tis)
            if x0.size > 1:
                fi.reinitialize_flow_field(layout_array=[xs0, ys0])
            else:
                fi.reinitialize_flow_field(layout_array=[xs, ys])
            fi.calculate_wake(yaw_angles=yaw_ini)
           # Get initial FLORIS power
            floris_power_0 = fi.get_farm_power()

            # Initialise FLORIS for optimal configuraiton
            fi.reinitialize_flow_field(layout_array=[xs, ys])
            fi.calculate_wake(yaw_angles=yws)
            # Get optimal FLORIS power
            floris_power_opt = fi.get_farm_power()

            floris_power_gain = round((floris_power_opt - floris_power_0)/floris_power_0*100, 2)

            if plots == True:
                print('|-----------------------|')
                print('Floris Initial Power', round(floris_power_0/1e6, 2), 'MW')
                print('Floris Optimal power', round(floris_power_opt/1e6, 2), 'MW')
                print('Floris Power Gain (%)', floris_power_gain)
                print('|-----------------------|')

            return -power_tot, floris_power_opt/1e6

        else:
            # Calculate power gain as provided by the DNN
            # (used in optimisation steps).

            return -power_tot
