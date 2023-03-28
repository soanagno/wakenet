from superposition import *
import floris


def florisOptimiser(
    ws,
    ti,
    layout_x,
    layout_y,
    min_yaw=-30,
    max_yaw=30,
    resx=dimx,
    resy=dimy,
    plots=False,
    mode="yaw",
    results=True
):
    """
    Calls the Floris optimiser to calculate the optimal yaws of a turbine farm.

    Args:
        ws (float) Downstream wind speed.
        ti (float) Downstream turbulence intesity.
        layout_x (numpy array of floats) Turbine x coordinates.
        layout_y (numpy array of floats) Turbine y coordinates.
        min_yaw (float) Minimum yaw of optimisation in degrees.
        max_yaw (float) Maximum yaw of optimisation in degrees.
        resx (float, optional) Horizontal resolution.
        resy (float, optional) Vertical resolution.
        plots (boolean, optional) If True, plots initial and optimised configuration.
        mode (boolean, optional) "yaw" for yaw optimisation, "farm" for layout optimisation.

    Returns:
        power_opt (float) Floris Optimised farm power output in MW.
            floris_time Floris optimisation time in seconds.
        power_initial (float) Floris initial farm power output in MW.
    """

    print()
    print()
    print("In FLORIS Optimiser...")

    # Instantiate the FLORIS object
    file_dir = os.path.dirname(os.path.abspath(__file__))
    fi = wfct.floris_interface.FlorisInterface(os.path.join(file_dir, file_path))

    # Initialise FLORIS wakefield
    if curl == True:
        fi.floris.farm.set_wake_model("curl")
    fi.reinitialize_flow_field(wind_speed=ws)
    fi.reinitialize_flow_field(turbulence_intensity=ti)
    fi.reinitialize_flow_field(layout_array=(layout_x, layout_y))
    fi.calculate_wake()
    power_initial = fi.get_farm_power()  # get initial power

    if mode == "yaw":

        # Start timer
        t0 = time.time()

        # Initialize the horizontal cut
        print(resx, resy)

        if plots == True:
            hor_plane = fi.get_hor_plane(height=hh, x_resolution=resx, y_resolution=resy)
            fig, ax = plt.subplots()
            wfct.visualization.visualize_cut_plane(hor_plane, ax=ax)
            ax.set_title("Baseline Case")

        opt_options = {
            "maxiter": 100,
            "disp": True,
            "iprint": 2,
            "ftol": 1e-7,
            "eps": 0.1,
        }

        # Instantiate the Optimization object
        yaw_opt = YawOptimization(
            fi,
            minimum_yaw_angle=min_yaw,
            maximum_yaw_angle=max_yaw,
            opt_options=opt_options,
        )

        # Perform optimization
        yaw_angles = yaw_opt.optimize()

        # Assign yaw angles to turbines and calculate wake
        fi.calculate_wake(yaw_angles=yaw_angles)
        power_opt = fi.get_farm_power()

        # End timer
        t1 = time.time()

        if results == True:
            print("==========================================")
            print("Inital Power = ", round(power_initial / 1e6, 2))
            print("Optimized Power = ", round(power_opt / 1e6, 2))
            print("Total Power Gain = %.1f%%" % (100 * (power_opt - power_initial) / power_initial))
            print("Floris Yaws: ", yaw_angles)
            print("==========================================")
        if plots == True:
            # Initialize the horizontal cut
            hor_plane = fi.get_hor_plane(height=hh, x_resolution=resx, y_resolution=resy)
            # Plot and show
            fig, ax = plt.subplots()
            wfct.visualization.visualize_cut_plane(hor_plane, ax=ax)
            ax.set_title("Optimal Wake Steering")
            plt.show()
        floris_time = round(t1 - t0, 2)
        print("FLORIS TIME:", floris_time)

        return power_opt / 1e6, floris_time, power_initial / 1e6

    elif mode == "farm":

        # Define turbine layout
        layout_x = list(layout_x)
        layout_y = list(layout_y)
        fi.reinitialize_flow_field(layout_array=(layout_x, layout_y))

        if plots == True:
            hor_plane = fi.get_hor_plane(height=hh,)
            fig, ax = plt.subplots()
            wfct.visualization.visualize_cut_plane(hor_plane, ax=ax)
            plt.show()

        t0 = time.time()

        boundaries = [
            [0, 0],
            [0, opt_ybound * D],
            [opt_xbound * D, opt_ybound * D],
            [opt_xbound * D, 0],
        ]

        # Generate random wind rose data 
        # (single wind direction and wind speed for this study)
        wd = np.array([270.0]*1)
        ws = np.ones(len(wd))*ws
        freq = np.abs(np.sort(np.random.randn(len(wd))))
        freq = freq / freq.sum()

        # Set optimization options
        opt_options = {"maxiter": 50, "disp": True, "iprint": 2, "ftol": 1e-8}

        # Compute initial AEP for optimization normalization
        AEP_initial = fi.get_farm_AEP(wd, ws, freq)

        # Instantiate the layout otpimization object
        layout_opt = LayoutOptimization(
            fi=fi,
            boundaries=boundaries,
            wd=wd,
            ws=ws,
            freq=freq,
            AEP_initial=AEP_initial,
            opt_options=opt_options,
        )

        # Perform layout optimization
        try:
            layout_results = layout_opt.optimize()
        except:
            layout_results = [layout_x, layout_y]
            file1 = open("opt_exeptions.txt", "a")
            file1.write(str(ws) + ' ' + str(ti) + " \n")
            file1.close()
            pass

        # Calculate new AEP results
        fi.reinitialize_flow_field(layout_array=(layout_results[0], layout_results[1]))
        AEP_optimized = fi.get_farm_AEP(wd, ws, freq)
        power_opt = fi.get_farm_power()

        if results == True:
            print("=====================================================")
            print("Total AEP Gain = %.1f%%" % (100.0 * (AEP_optimized - AEP_initial) / AEP_initial))
            print("Floris Initial Power", round(power_initial / 1e6, 2))
            print("Floris Optimal Power", round(power_opt / 1e6, 2))
            print("Total Power Gain (%)", round((power_opt - power_initial)/power_initial * 100, 2))
            print("Floris Layout: ", layout_results)
            print("=====================================================")
        t1 = time.time()
        floris_time = round(t1 - t0, 2)
        print("FLORIS TIME:", floris_time)
        if plots == True:
            # Plot the new layout vs the old layout
            layout_opt.plot_layout_opt_results()
            plt.show()
            hor_plane = fi.get_hor_plane(height=hh,)
            fig, ax = plt.subplots()
            wfct.visualization.visualize_cut_plane(hor_plane, ax=ax)
            plt.show()

        return power_opt / 1e6, floris_time, power_initial / 1e6


def _norm(val, x1, x2):
    return (val - x1) / (x2 - x1)


def _space_constraint(x_in):
    min_dist = 2.0*D

    nturbs = int(x_in.size / 2.0 + 0.25)
    x = np.array(x_in[:nturbs])
    y = np.array(x_in[nturbs:])

    dist = [
        np.sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2)
        for i in range(nturbs)
        for j in range(nturbs)
        if i != j
    ]

    dist = np.array(dist)

    # Normalise boundaries (from Floris)
    bndx_min = 0.0
    bndx_max = (opt_xbound+1.0)*D

    an = _norm(np.min(dist), bndx_min, bndx_max)
    bn = _norm(min_dist, bndx_min, bndx_max)

    return an - bn


def neuralOptimiser(
    ws,
    ti,
    xs,
    ys,
    min_yaw=-30,
    max_yaw=30,
    plots=False,
    plots_ini=False,
    floris_gain=False,
    mode="yaw",
    results=True
):

    """
    Calls the Floris optimiser to calculate the optimal yaws of a turbine farm.

    Args:
        ws (float) Downstream wind speed.
        ti (float) Downstream turbulence intesity.
        xs (numpy array of floats) Turbine x coordinates.
        ys (numpy array of floats) Turbine y coordinates.
        min_yaw (float) Minimum yaw of optimisation in degrees.
        max_yaw (float) Maximum yaw of optimisation in degrees.
        plots (boolean, optional) If True, plots initial and optimised configuration.
        floris_gain (boolean, optional) If True, calculates and returns gained power
            output with the optimised results of the DNN but using Floris for comparison.
        mode ('yaw' or 'farm') Specifies which optimisation mode is to be run.

    Returns:
        floris_power_opt (float) Total farm power output produced by Floris in MW,
            based on the input turine yaws and positions produced by the neural optimiser.
        neural_time (float) Neural Network optimisation time in seconds.
    """

    print()
    print()
    print("In NEURAL Optimiser...")

    layout = np.concatenate((xs, ys), axis=0)

    if mode == "yaw":

        # Calculate initial power
        power_ini = -superposition(
            np.zeros(xs.size),
            layout,
            u_stream=ws,
            tis=ti,
            cp=cp,
            wind_speed=wind_speed,
            plots=plots_ini,
            power_opt=True,
        )

        # Optimiser options
        opt_options = {
            "maxiter": 100,
            "disp": True,
            "ftol": 1e-7,
            "eps": 0.1,
        }

        # Set initial yaws
        x0 = (yaw_ini,) * xs.size
        # Set min-max yaw constraints
        bnds = ((min_yaw, max_yaw),) * xs.size

        # Optimise and time
        t0 = time.time()
        res = minimize(
            superposition,
            x0,
            args=(layout, ws, ti, cp, wind_speed),
            method="SLSQP",
            bounds=bnds,
            options=opt_options,
        )
        t1 = time.time()
        neural_time = round(t1 - t0, 2)

        optimal, floris_power_opt, floris_power_0 = superposition(
            res.x,
            layout,
            u_stream=ws,
            tis=ti,
            cp=cp,
            wind_speed=wind_speed,
            plots=plots,
            power_opt=True,
            floris_gain=floris_gain,
        )

        if results == True:
            print("-----------------------------------------------------")
            print("Floris Initial Power", round(floris_power_0, 2), "MW")
            print("Floris Optimal Power", round(floris_power_opt, 2), "MW")
            print("Floris Power Gain (%)", round((np.abs(floris_power_opt) - floris_power_0)/floris_power_0 * 100, 2))
            print("Neural Yaws:", np.round(res.x, 2))
            print("-----------------------------------------------------")
        print("NEURAL TIME:", neural_time)

        return floris_power_opt, neural_time, floris_power_0

    elif mode == "farm":

        farm_opt = True
        x0 = np.copy(layout)  # Save initial layout positions

        # Define minimum distance between turbines
        min_dist = 2.0 * D
        tmp1 = {
            "type": "ineq",
            "fun": lambda x, *args: _space_constraint(x, min_dist),
            "args": (min_dist,),
        }
        con = {'type': 'ineq', 'fun': _space_constraint}

        # Optimiser options
        opt_options = {"maxiter": 100, "disp": True, "iprint": 2, "ftol": 1e-8, "eps": 5}
        # Set initial yaws
        yws = (yaw_ini,) * xs.size
        # Set min-max boundary constraints (+1D to match FLORIS bnds)
        bnds = ((0, opt_xbound*D),)*xs.size + ((0, opt_ybound*D),)*xs.size

        # Optimise and time
        t0 = time.time()
        res = minimize(
            superposition,
            layout,
            args=(yws, ws, ti, cp, wind_speed, farm_opt),
            method="SLSQP",
            bounds=bnds,
            options=opt_options,
            constraints=con,
        )
        t1 = time.time()

        neural_time = round(t1 - t0, 2)

        optimal, floris_power_opt, floris_power_0 = superposition(
            res.x,
            yws,
            u_stream=ws,
            tis=ti,
            cp=cp,
            wind_speed=wind_speed,
            farm_opt=farm_opt,
            plots=plots,
            power_opt=True,
            floris_gain=floris_gain,
            x0=x0,
        )

        if results == True:
            print("-----------------------------------------------------")
            print("Floris Initial Power", round(floris_power_0, 2), "MW")
            print("Floris Optimal Power", round(floris_power_opt, 2), "MW")
            print("Floris Power Gain (%)", round((np.abs(floris_power_opt) - floris_power_0)/floris_power_0 * 100, 2))
            print("Neural Layout:", np.round(res.x, 2))
            print("-----------------------------------------------------")
        print("NEURAL TIME:", neural_time)

        # return floris_power_gain, neural_time
        return floris_power_opt, neural_time, floris_power_0


def compare(
    yws,
    ws,
    ti,
    xs,
    ys,
    plots=False,
    print_times=True,
    timings=False,
    power_opt=True,
    single=False,
    saveas=None,
):
    """
    Performs a comparison between a wind farm produced
    by the Neural Network vs Floris.

    Args:
        yws (numpy array of floats) Yaws of each turbine.
        ws (float) Free stream velocity.
        ti (floats) Turbulence intensity.
        xs (numpy array of floats) Turbine x coordinates.
        ys (numpy array of floats) Turbine y coordinates.
        print_times (boolean, optional) If True, prints timings.
        single (boolen, optional) If True, calculates single turbine

    """

    f = open(file_path,)
    data = json.load(f)
    f.close()

    layout = np.concatenate((xs, ys), axis=0)
    cp = np.array(data["turbine"]["properties"]["power_thrust_table"]["power"])
    wind_speed = np.array(
        data["turbine"]["properties"]["power_thrust_table"]["wind_speed"]
    )

    return superposition(
        yws,
        layout,
        u_stream=ws,
        tis=ti,
        cp=cp,
        wind_speed=wind_speed,
        plots=plots,
        power_opt=power_opt,
        print_times=print_times,
        timings=timings,
        floris_gain=True,
        single=single,
        saveas=saveas,
    )


def heatmap(xs, ys, res=10, farm_opt=False, saveas=None):
    """
    Assess the performance of the DNN vs FLORIS on
    parametric optimiser calls for a wide range of
    inlet speed and turbulence intensity for a
    specific array configuration.

    Args:
        xs (numpy array of floats) Turbine x coordinates.
        ys (numpy array of floats) Turbine y coordinates.
        res (int, optional) Resolution of heatmap.
        farm_opt (boolean, optional) Calls either farm or yaw optimisers.
    """

    # Wind speeds and turbulence intensities examined
    x_ws = np.linspace(ws_range[0], ws_range[1], res)
    y_ti = np.linspace(ti_range[0], ti_range[1], res)
    if res == 3:
        x_ws = np.array([11, 12.33333333, 13.66666667])
        y_ti = np.array([0.01, 0.03111111, 0.05222222])

    # Initialisation of power and timing heatmaps
    g0 = np.zeros((res, res))
    g1 = np.zeros((res, res))
    g2 = np.zeros((res, res))
    t1 = np.zeros((res, res))
    t2 = np.zeros((res, res))

    only_ddn = False
    cnt = 0
    # Begin parametric runs
    for k1 in range(res):

        for k2 in range(res):

            # Print progress
            print()
            print('OPTIMISATION PROGRESS:', int(cnt/res/res*100), "%", "COMPLETE.")
            print()
            cnt+=1

            if farm_opt == True:
                if only_ddn == False:
                    g1[k1, k2], t1[k1, k2], g0[k1, k2] = florisOptimiser(
                        ws=x_ws[k1], 
                        ti=y_ti[k2],
                        layout_x=xs, 
                        layout_y=ys,
                        mode="farm"
                    )
                g2[k1, k2], t2[k1, k2], g0[k1, k2] = neuralOptimiser(
                    ws=x_ws[k1],
                    ti=y_ti[k2],
                    xs=xs,
                    ys=ys,
                    floris_gain=True,
                    mode="farm",
                )
            else:
                if only_ddn == False:
                    g1[k1, k2], t1[k1, k2], g0[k1, k2] = florisOptimiser(
                        ws=x_ws[k1],
                        ti=y_ti[k2],
                        layout_x=xs,
                        layout_y=ys
                    )
                g2[k1, k2], t2[k1, k2], g0[k1, k2]= neuralOptimiser(
                    ws=x_ws[k1],
                    ti=y_ti[k2],
                    xs=xs, ys=ys,
                    floris_gain=True
                )

    if saveas != None:
        save1 = saveas+"floris_opt"; save2=saveas+"ddn_opt"; save3=saveas+"floris_t"; save4=saveas+"ddn_t"
    else:
        save1, save2, save3, save4 = saveas, saveas, saveas, saveas

    # Calculate FLORIS power gain in MW
    sample = g1 - g0
    mv = makeHeatmap(
        np.transpose(np.flip(sample, 1)), x_ws, y_ti, title="Floris optimisation", saveas=save1
    )
    if only_ddn == True:
        # mval = None
        mval = 2.1
    else:
        mval = mv
    # Calculate DNN power gain in MW
    sample = g2 - g0
    makeHeatmap(
        np.transpose(np.flip(sample, 1)), x_ws, y_ti, mval, title="Neural optimisation", saveas=save2
    )
    # Calculate FLORIS average time
    sample = t1
    print("Average FLORIS time:", np.round(np.mean(t1), 2))
    mv = makeHeatmap(np.transpose(np.flip(sample, 1)), x_ws, y_ti, title="Floris time", saveas=save3
    )
    if only_ddn == True:
        # mval = None
        mval = 1700
    else:
        mval = mv
    # Calculate DNN average time
    sample = t2
    print("Average DNN time:", np.round(np.mean(t2), 2))
    makeHeatmap(np.transpose(np.flip(sample, 1)), x_ws, y_ti, mval, title="Neural time", saveas=save4
    )


def makeHeatmap(bitmap, x_ws, y_ti, vmax=None, title=None, saveas=None):
    """
    Plots bitmap of parametric optimisation runs.

    Args:
        bitmap (2D numpy array of floats) Calculated powers.
        x_ws (1D numpy array of floats) Wind speeds.
        y_ti (1D numpy array of floats) Turbulence intensities.
        vmax (float, optional) Max velocity cap of plot.
        title (string) Plot title.

    """

    # Min and max values of heatmap
    x_min = np.min(x_ws)
    x_max = np.max(x_ws)
    y_min = np.min(y_ti)
    y_max = np.max(y_ti)

    maxval = np.max(np.abs([bitmap.min(), bitmap.max()]))
    if vmax:
        maxval = vmax
    vmin = -maxval

    if title == "Floris time" or title == "Neural time":
        cmap = "RdYlGn_r"
        vmin = 0
    else:
        cmap = "RdYlGn"

    # Plot heatmap based on bitmap produced by the "Assess" function.
    plt.figure()
    plt.imshow(
        bitmap,
        cmap=cmap,
        interpolation="nearest",
        vmin=vmin,
        vmax=maxval,
        extent=[x_min, x_max, y_min, y_max],
        aspect=(x_max - x_min) / (y_max - y_min),
    )

    plt.title(title, fontname="serif")
    plt.xlabel("Free stream velocity (m/s)", fontname="serif")
    plt.ylabel("Turbulence intensity", fontname="serif")

    plt.colorbar()
    if saveas != None:
        plt.savefig("figures/"+str(saveas), dpi=1200)
    else:
        plt.show()

    return maxval


def yawVsPowerContour(yws, ws, ti, xs, ys, res=30, saveas=None):
    """
    Plot 2 turbine wind farm yaw-power contour
    """

    from mpl_toolkits import mplot3d

    x = np.linspace(0, res, res)
    y = np.linspace(0, res, res)

    X, Y = np.meshgrid(x, y)

    powerNeural = np.zeros((res, res))
    powerFloris = np.zeros((res, res))
    cnt = 0
    for i in range(res):
        for j in range(res):

            if len(yws) == 2:
                yws = [i, j]
            elif len(yws) == 3:
                yws = [0, i, j]

            r = compare(
                yws,
                ws,
                ti,
                xs,
                ys,
                print_times=False,
                timings=False,
                power_opt=True,
                single=False,
            )

            powerNeural[i, j], powerFloris[i, j] = -r[0] / 1e6, r[1]

    fig = plt.figure(1)
    ax = plt.axes(projection="3d")
    ax.contour3D(X, Y, powerNeural, 50, cmap="viridis")
    # ax.plot_surface(X, Y, powerNeural, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax.set_xlabel("Yaw b")
    ax.set_ylabel("Yaw a")
    ax.set_zlabel("Power (MW)")
    ax.set_title("Neural")

    if saveas != None:
        fig.savefig("figures/"+str(saveas)+"yvpd", dpi=1200)
    else:
        plt.show()

    fig = plt.figure(2)
    ax = plt.axes(projection="3d")
    ax.contour3D(X, Y, powerFloris, 50, cmap="viridis")
    ax.set_xlabel("Yaw b")
    ax.set_ylabel("Yaw a")
    ax.set_zlabel("Power (MW)")
    ax.set_title("FLORIS")

    if saveas != None:
        fig.savefig("figures/"+str(saveas)+"yvpf", dpi=1200)
    else:
        plt.show()
