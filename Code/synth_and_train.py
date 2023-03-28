from neuralWake import *


def set_seed(seed):
    """
    Use this to set ALL the random seeds to a fixed value and remove randomness from cuda kernels
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Uses the inbuilt cudnn auto-tuner to find the fastest convolution algorithms
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

    return True


def normalise(x, mode, print_output=False):
    """
    Normalises input data.

    Args:
        x (numpy float array) Data to be normalised.
        mode (values: 1, 2, 3) Modes of normalisation,
            1: Min-Max [0, 1], 2: Z-Score [-0.5, 0.5], 3: Min-Max [-1, 1].
        print_output (boolean, optional) Prints normalised data.

    Returns:
        x (numpy float array) Normalised data.
    """

    if mode == 1:
        x = np.true_divide(x - np.min(x), np.max(x) - np.min(x))

        if print_output == True:
            print("Normalised speeds:", x)
            input("Press enter to continue...")
    elif mode == 2:
        x -= np.mean(x)
        x /= np.std(x)

        if print_output == True:
            print("Normalised speeds:", x)
            input("Press enter to continue...")
    elif mode == 3:
        x = (np.true_divide(x - np.min(x), np.max(x) - np.min(x)) - 0.5) * 2

        if print_output == True:
            print("Normalised speeds:", x)
            input("enter")
    return x


def create(plots=False):
    """
    Generates synthetic wake deficit data.

    Args:
        plots (boolean, optional) Plots indicative sample.

    Returns:
        X_train, X_val, X_test (1D numpy float arrays) Training, validation
            and test sets input inlet conditions.
        y_train, y_val, y_test (1D numpy float arrays) Training, validation
            and test sets output wake deficits as calculated by Floris.
    """

    # Random Dataset
    speeds, ti = wakeNet.tiVsVel(data_size)
    np.random.seed(51)
    yw = (np.random.rand(data_size) - 0.5) * (
        yw_range[1] - yw_range[0]
    )  # hub yaw angles
    np.random.seed(256)
    hbs = (
        np.random.rand(data_size) * (hb_range[1] - hb_range[0]) + hb_range[0]
    )  # height slice

    print("Max inlet speed:", round(np.max(speeds), 2), "m/s")


    speeds_out = np.zeros((data_size, out_piece, rows))
    u_rs = np.zeros((out_piece, rows))

    sample_plots = []
    cnt = 1
    sample_size = 9  # must be perfect square for sample plots

    if save_data == True:
        print("Are you sure you want to create new dataset? (y/n)")
        if input() == "y":
            os.system("mkdir " + "wake_dataset")
            np.save("wake_dataset/inlets.npy", np.stack((speeds, ti, yw), axis = 1))
    elif curl == True:
        inlets = np.load("wake_dataset/inlets.npy")
        speeds, ti, yw = inlets[:data_size, 0], inlets[:data_size, 1], inlets[:data_size, 2]

    for i in range(data_size):

        if curl == True:
            fi.floris.farm.set_wake_model("curl")

        if make_data == True:

            if i == 0:
                print("Synthesizing data...")

            if i % 100 == 0:
                print("Synthesised", int(i / data_size * 100), "%", "of wakes.")

            if inputs == 1:
                fi.reinitialize_flow_field(wind_speed=speeds[i])
                fi.calculate_wake()
            if inputs == 2:
                fi.reinitialize_flow_field(wind_speed=speeds[i])
                fi.reinitialize_flow_field(turbulence_intensity=ti[i])
                fi.calculate_wake()
            if inputs == 3:
                fi.reinitialize_flow_field(wind_speed=speeds[i])
                fi.reinitialize_flow_field(turbulence_intensity=ti[i])
                fi.calculate_wake(yaw_angles=yw[i])
            if inputs == 4:
                fi.reinitialize_flow_field(wind_speed=speeds[i])
                fi.reinitialize_flow_field(turbulence_intensity=ti[i])
                fi.change_turbine([0], {"yaw_angle": yw[i]})
                cut_plane = fi.get_hor_plane(
                    height=hbs[i],
                    x_resolution=dimx,
                    y_resolution=dimy,
                    x_bounds=x_bounds,
                    y_bounds=y_bounds,
                )
            else:
                cut_plane = fi.get_hor_plane(
                    height=hh,
                    x_resolution=dimx,
                    y_resolution=dimy,
                    x_bounds=x_bounds,
                    y_bounds=y_bounds,
                )

            u_mesh = cut_plane.df.u.values.reshape(
                cut_plane.resolution[1], cut_plane.resolution[0]
            )

        if save_data == True:

            # Save velocities as numpy array
            np.save("wake_dataset/" + "wake" + str(i), u_mesh)

            continue

        if save_data == False and curl == True:

            if i == 0:
                print("Loading data...")

            if i % 100 == 0:
                print("Loaded ", int(i / data_size * 100), "%", "of wakes.")

            # Read back into different array "r"
            u_mesh = np.load("wake_dataset/" + "wake" + str(i) + ".npy")

        if row_major == 0:
            u_mesh = u_mesh.T

        u_mesh = u_mesh.flatten()

        for kapa in range(rows):
            u_rs[:, kapa] = u_mesh[kapa * out_piece : (kapa + 1) * out_piece]

        if cubes == 1:
            jj = 0
            ii = 0
            alpha = np.zeros((dim1 * dim2, int(u_rs.size / (out_piece))))

            for k in range(int(u_rs.size / (dim1 * dim2))):

                alpha[:, k] = u_rs[ii : ii + dim1, jj : jj + dim2].flatten("C")

                jj += dim2
                if jj >= u_rs.shape[1]:
                    jj = 0
                    ii += dim1

            speeds_out[i] = alpha
        else:
            speeds_out[i] = u_rs

        # Store synthesized data for plotting
        if plots == True:
            sample_plots.append(cut_plane)

        # Plot synthesized data (batches of sample_size)
        if plots == True and np.mod(i + 1, sample_size) == 0:

            fig, axarr = plt.subplots(
                int(np.sqrt(sample_size)),
                int(np.sqrt(sample_size)),
                sharex=True,
                sharey=True,
                figsize=(12, 5),
            )
            axarr = axarr.flatten()

            minspeed = np.min(speeds[(cnt - 1) * sample_size : cnt * sample_size])
            maxspeed = np.max(speeds[(cnt - 1) * sample_size : cnt * sample_size])

            for ii in range(sample_size):

                ax = axarr[ii]
                title = (
                    "("
                    + str(np.round(speeds[(cnt - 1) * sample_size + ii], 1))
                    + ", "
                    + str(np.round(ti[(cnt - 1) * sample_size + ii], 2))
                    + ", "
                    + str(np.round(yw[(cnt - 1) * sample_size + ii], 1))
                    + ")"
                )

                hor_plane = sample_plots[ii]
                wfct.visualization.visualize_cut_plane(
                    hor_plane, ax=ax, minSpeed=minspeed, maxSpeed=maxspeed
                )
                ax.set_title(title)

                ax.set_yticklabels(ax.get_yticks().astype(int))
                ax.set_xticklabels(ax.get_xticks().astype(int))

            plt.show()
            sample_plots = []
            cnt += 1

    # Normalisation
    speeds = ((speeds - ws_range[0]) / (ws_range[1] - ws_range[0]) - 0.5) * 3
    ti = ((ti - ti_range[0]) / (ti_range[1] - ti_range[0]) - 0.5) * 3
    yw = ((yw - yw_range[0]) / (yw_range[1] - yw_range[0]) - 0.5) * 3
    hbs = ((hbs - hb_range[0]) / (hb_range[1] - hb_range[0]) - 0.5) * 3

    # Make X and y
    X_input = np.zeros((data_size, inputs))

    if inputs == 1:
        X_input[:, 0] = speeds
    if inputs == 2:
        X_input[:, 0] = speeds
        X_input[:, 1] = ti
    if inputs == 3:
        X_input[:, 0] = speeds
        X_input[:, 1] = ti
        X_input[:, 2] = yw
    if inputs == 4:
        X_input[:, 0] = speeds
        X_input[:, 1] = ti
        X_input[:, 2] = yw
        X_input[:, 3] = hbs

    X = torch.tensor(X_input, dtype=torch.float)
    y = torch.tensor(speeds_out, dtype=torch.float)

    X = X.view(data_size, -1)
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    # Train, Validation, Test slices
    c1 = int(data_size * (train_slice))
    c2 = int(data_size * (train_slice + val_slice))
    c3 = int(data_size * (train_slice + val_slice + test_slice))

    X_train = X[:c1]
    y_train = y[:c1]
    X_val = X[c1:c2]
    y_val = y[c1:c2]
    X_test = X[c2:c3]
    y_test = y[c2:c3]

    return X_train, X_val, X_test, y_train, y_val, y_test


def dif_central(u, dx, eflag=0):

    batches = u.shape[0]
    u_x = torch.ones_like(u)

    for ii in range(batches):
        for jj in range(1, dimx-1):
            u_x[ii, :, jj] = (u[ii, :, jj+1] - u[ii, :, jj-1])/(2*dx)
        u_x[ii, :, 0] = (u[ii, :, 1] - u[ii, :, 0])/dx
        u_x[ii, :, -1] = (u[ii, :, -2] - u[ii, :, -1])/dx
    
    if eflag==-1:
        u_x[:,:,:10] = 0
        plt.figure(1)
        plt.imshow(u[0].detach().cpu().numpy())
        plt.figure(2)
        plt.imshow(u_x[0].detach().cpu().numpy())
        plt.show()

    return u_x


def training(X_train, X_val, X_test, y_train, y_val, y_test, model, plot_curves=0,
            multiplots=False, data_size=data_size, batch_size=batch_size, saveas=None):
    """
    Trains the neural model.

    Args:
        plots (boolean, optional) Plots indicative sample.
    """

    if batch_size > X_train.shape[0]:
        print('Error: batch_size must be <', X_train.shape[0])
        exit()

    # Define validation and test batch sizes
    val_batch_size = y_val.size()[0]
    train_split = TensorDataset(X_train, y_train[:, :, -1])
    validation_split = TensorDataset(X_val, y_val[:, :, -1])
    train_loader = DataLoader(
        train_split, batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True
    )
    validation_loader = DataLoader(
        validation_split, batch_size=val_batch_size, shuffle=True, num_workers=workers, drop_last=True
    )

    #  Seed, optimiser and criterion
    set_seed(42)
    params = list(model.fc1.parameters()) + \
             list(model.fc2.parameters()) + \
             list(model.fc3.parameters())
    # Optimizers
    if opt_method == "SGD":
        optimizer = optim.SGD(params, lr=lr, momentum=momentum)
    elif opt_method == "Rprop":
        optimizer = optim.Rprop(params, lr=lr, etas=(0.5, 1.2), step_sizes=(1e-06, 50))
    elif opt_method == "Adam":
        optimizer = optim.Adam(params, lr=lr)
    # Loss criterions
    criterion = nn.MSELoss(size_average=1)
    criterion = criterion.to(device)

    # Initialise plots
    t_plot = []; v_plot = []
    t_loss_plot = []; v_loss_plot = []
    lossmin = 1e16; valmax = 0.5

    # Model Training
    for i_epoch in range(epochs):

        print("Epoch:", i_epoch, "/", epochs)
        t_loss = 0; t_lossc1 = 0; t_lossc1_ = 0; t_lossc2 = 0; t_acc = 0
        v_loss = 0; v_acc = 0; v_lossc1_ = 0; v_min = 0;

        model.train().to(device)
        eflag = i_epoch
        for X, y in train_loader:
            # Get yt_pred
            X, y = X.to(device), y.to(device)
            yt_pred = model(X)
            c1 = criterion(yt_pred, y)
            yy = yt_pred.detach().cpu().numpy()
            yy_ = y.detach().cpu().numpy()
            c2 = torch.tensor(0)

            # Losses
            train_loss = c1 + c2
            t_loss += train_loss.item()
            tterm = torch.abs(y - yt_pred)/torch.max(y)
            t_acc += torch.sum(torch.pow(tterm, 2)).detach().cpu().numpy()
            t_lossc1 += c1.item()
            t_lossc1_ += torch.sum(torch.pow(y - yt_pred, 2)).detach().cpu().numpy()
            t_lossc2 += c2.item()

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            eflag = 0

        # Training results
        t_loss = t_loss/(train_slice*data_size/batch_size)
        t_lossc1 = t_lossc1/(train_slice*data_size/batch_size)
        t_lossc1_ /= train_slice*data_size*out_piece
        t_lossc2 = t_lossc2/(train_slice*data_size/batch_size)
        t_acc /= train_slice*data_size*out_piece
        t_acc = 1 - np.sqrt(t_acc)

        model.eval().to(device)
        for X, y in validation_loader:
            with torch.no_grad():

                val_batch = y.shape[0]
                X, y = X.to(device), y.to(device)
                y_pred = model(X)
                c1 = criterion(y_pred, y)
                c2 = torch.tensor(0)

                val_loss = c1 + c2
                v_loss += val_loss.item()
                vvterm = torch.abs(y - y_pred)/torch.max(y)
                v_acc += torch.sum(torch.pow(vvterm, 2)).detach().cpu().numpy()
                v_min += torch.min(1 - torch.abs(y - y_pred)).detach().cpu().numpy()
                v_lossc1_ += torch.sum(torch.pow(y - y_pred, 2)).detach().cpu().numpy()

        # # Validation results
        v_loss = v_loss/(val_batch_size/val_batch)
        v_lossc1_ /= val_batch_size*out_piece
        v_acc /= val_batch_size*out_piece
        v_acc = 1 - np.sqrt(v_acc)
        v_min /= val_batch_size*out_piece

        # Append to plots
        t_plot.append(t_acc); v_plot.append(v_acc)
        t_loss_plot.append(t_loss); v_loss_plot.append(v_loss)

        if v_loss < lossmin: # and i_epoch > epochs*0.8:
            lossmin = v_loss
            # Save model weights
            torch.save(model.state_dict(), weights_path)
            print("Saved weights with", v_loss, "loss")
        if v_acc > valmax: # and i_epoch > epochs*0.8:
            valmax = v_acc

        # Mean sum squared loss
        print(
              "t_acc: " + str(round(t_acc, 4)) + " v_acc: " + str(round(v_acc, 4))
            + " t_loss: " + str(round(t_loss, 2)) +  " v_loss: " + str(round(v_loss, 2))
            + " t_lossc1: " + str(round(t_lossc1, 2)) + " t_lossc2: " + str(round(t_lossc2, 2))
        )

    # ------------- Loss and Accuracy Plots -------------#
    if plot_curves == 1 or saveas != None:

        fig, axs = plt.subplots(1, 2)
        del fig

        axs[0].plot(np.arange(epochs), t_loss_plot, color="navy", linestyle="--")
        axs[0].plot(np.arange(epochs), v_loss_plot, color="crimson")
        axs[1].plot(np.arange(epochs), t_plot, color="navy", linestyle="--")
        axs[1].plot(np.arange(epochs), v_plot, color="crimson")
        axs[1].set_ylim(0.5, 1)

        print("Validation loss:", lossmin)
        print("Validation accuracy:", valmax)

        axs[0].tick_params(axis="x", direction="in")
        axs[0].tick_params(axis="y", direction="in")
        axs[0].set_aspect(aspect=1.0 / axs[0].get_data_ratio())
        axs[1].tick_params(axis="x", direction="in")
        axs[1].tick_params(axis="y", direction="in")
        axs[1].set_aspect(aspect=1.0 / axs[1].get_data_ratio())

        if saveas != None:
            plt.savefig("figures/"+str(saveas), dpi=1200)
        elif multiplots == False:
            plt.show()

    # Replace last values with best values
    v_loss_plot[-1] = lossmin
    v_plot[-1] = valmax

    return v_loss_plot, t_loss_plot, v_plot, t_plot
