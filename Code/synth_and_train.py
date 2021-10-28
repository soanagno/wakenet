from neuralWake import *

# Set GPU if Available
if device == 'gpu':
    if torch.cuda.device_count() > 0 and torch.cuda.is_available():
        print("Cuda installed! Running on GPU!")
        device = 'cuda'
    else:
        device = 'cpu'
        print("No GPU available! Running on CPU.")


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
    torch.backends.cudnn.enabled   = False

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
            print('Normalised speeds:', x)
            input('Press enter to continue...')
    elif mode == 2:
        x -= np.mean(x)
        x /= np.std(x)

        if print_output == True:
            print('Normalised speeds:', x)
            input('Press enter to continue...')
    elif mode == 3:
        x = (np.true_divide(x - np.min(x), np.max(x) - np.min(x)) - 0.5)*2

        if print_output == True:
            print('Normalised speeds:', x)
            input('enter')
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

    print('Synthesizing data...')

    # Random Dataset
    speeds, ti = wakeNet.tiVsVel(data_size)
    np.random.seed(51)
    yw = (np.random.rand(data_size) - 0.5)*(yw_range[1] - yw_range[0])         # hub yaw angles
    np.random.seed(256)
    hbs = np.random.rand(data_size)*(hb_range[1] - hb_range[0]) + hb_range[0]  # height slice

    print('Max inlet speed:', round(np.max(speeds), 2), 'm/s')

    speeds_out = np.zeros((data_size, out_piece, rows))
    u_rs = np.zeros((out_piece, rows))

    sample_plots = []
    cnt = 1
    sample_size = 9  # must be perfect square for sample plots

    for i in range(data_size):
        
        if i%100 == 0:
            print('Synthesised ', int(i/data_size*100), '%', 'of wakes.')
        # fi.floris.farm.set_wake_model('curl')

        if inputs == 1:
            fi.reinitialize_flow_field(wind_speed = speeds[i])
            fi.calculate_wake()
        if inputs == 2:
            fi.reinitialize_flow_field(wind_speed = speeds[i])
            fi.reinitialize_flow_field(turbulence_intensity = ti[i])
            fi.calculate_wake()

        if inputs == 3:
            fi.reinitialize_flow_field(wind_speed = speeds[i])
            fi.reinitialize_flow_field(turbulence_intensity = ti[i])
            fi.calculate_wake(yaw_angles=yw[i])

        if inputs == 4:
            fi.reinitialize_flow_field(wind_speed = speeds[i])
            fi.reinitialize_flow_field(turbulence_intensity = ti[i])
            fi.change_turbine([0],{'yaw_angle':yw[i]})
            cut_plane = fi.get_hor_plane(height=hbs[i],
                                         x_resolution=dimx,
                                         y_resolution=dimy,
                                         x_bounds=x_bounds,
                                         y_bounds=y_bounds)
        else:
            cut_plane = fi.get_hor_plane(height=hh,
                                         x_resolution=dimx,
                                         y_resolution=dimy,
                                         x_bounds=x_bounds,
                                         y_bounds=y_bounds)

        u_mesh = cut_plane.df.u.values.reshape(cut_plane.resolution[1],
                                               cut_plane.resolution[0])

        if row_major == 0:
            u_mesh = u_mesh.T

        u_mesh = u_mesh.flatten()

        for kapa in range(rows):
            u_rs[:, kapa] = u_mesh[kapa*out_piece:(kapa+1)*out_piece]

        if cubes == 1:
            jj = 0; ii = 0
            alpha = np.zeros((dim1*dim2, int(u_rs.size/(out_piece))))

            for k in range(int(u_rs.size/(dim1*dim2))):

                alpha[:, k] = u_rs[ii:ii+dim1, jj:jj+dim2].flatten('C')

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
        if plots == True and np.mod(i+1, sample_size) == 0:

            sizeOfFont = 11
            fig, axarr = plt.subplots(int(np.sqrt(sample_size)), int(np.sqrt(sample_size)),
                                      sharex=True, sharey=True, figsize=(12, 5))
            axarr = axarr.flatten()

            minspeed = np.min(speeds[(cnt-1)*sample_size : cnt*sample_size])
            maxspeed = np.max(speeds[(cnt-1)*sample_size : cnt*sample_size])

            for ii in range(sample_size):

                ax = axarr[ii]
                title = '(' + str(np.round(speeds[(cnt-1)*sample_size + ii], 1)) + ', ' + \
                            str(np.round(ti[(cnt-1)*sample_size + ii], 2)) + ', ' + \
                            str(np.round(yw[(cnt-1)*sample_size + ii], 1)) + ')'

                hor_plane = sample_plots[ii]
                wfct.visualization.visualize_cut_plane(hor_plane, ax=ax,
                                                       minSpeed=minspeed, maxSpeed=maxspeed)
                ax.set_title(title, fontname='serif', fontsize=sizeOfFont)

                fontProperties = {'family':'serif',
                    'weight' : 'normal', 'size' : sizeOfFont}

                ax.set_yticklabels(ax.get_yticks().astype(int), fontProperties)
                ax.set_xticklabels(ax.get_xticks().astype(int), fontProperties)
                
            plt.show()
            sample_plots = []
            cnt += 1

    # Apply normalisations on inputs
    yw = normalise(yw, norm)
    ti = normalise(ti, norm)
    speeds = normalise(speeds, norm)
    hbs = normalise(hbs, norm)

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

    X = torch.tensor(X_input, dtype = torch.float)
    y = torch.tensor(speeds_out, dtype = torch.float)

    X = X.view(data_size, -1)
    print('X shape:', X.shape)
    print('y shape:', y.shape)

    # Train, Validation, Test slices
    c1 = int(data_size * (train_slice))
    c2 = int(data_size * (train_slice + val_slice))
    c3 = int(data_size * (train_slice + val_slice + test_slice))

    X_train = X[:c1]; y_train = y[:c1]
    X_val = X[c1:c2]; y_val = y[c1:c2]
    X_test = X[c2:c3]; y_test = y[c2:c3]

    return X_train, X_val, X_test, y_train, y_val, y_test


def training(ii,
             X_train,
             X_val,
             X_test,
             y_train,
             y_val,
             y_test,
             model):
    """
    Trains the neural model.

    Args:
        plots (boolean, optional) Plots indicative sample.
    """

    # Define validation and test batch sizes
    val_batch_size = y_val.size()[1]
    # test_batch_size = y_test.size()[1]

    train_split = TensorDataset(X_train, y_train[:, :, ii])
    validation_split = TensorDataset(X_val, y_val[:, :, ii])
    train_loader = DataLoader(train_split, batch_size=batch_size, shuffle=True, num_workers=workers)
    validation_loader = DataLoader(validation_split, batch_size=val_batch_size,
                                   shuffle=True, num_workers=workers)

    #  Seed, optimiser and criterion
    set_seed(42)

    # Parameters to optimize
    # params = list(model.fc1[ii].parameters()) + list(model.fc2[ii].parameters()) + \
    #          list(model.fc3[ii].parameters()) + list(model.fc4[ii].parameters())
    params = list(model.fc1[ii].parameters()) + \
             list(model.fc2[ii].parameters()) + \
             list(model.fc3[ii].parameters())
    # params = list(model.fc1[ii].parameters()) + list(model.fc3[ii].parameters())

    # Optimizers
    if opt_method == 'SGD':
        optimizer = optim.SGD(params, lr=lr, momentum=momentum) # , weight_decay=weight_decay)
    elif opt_method == 'Rprop':
        optimizer = optim.Rprop(params, lr=lr, etas=(0.5, 1.2), step_sizes=(1e-06, 50))

    # Loss criterions
    criterion = nn.MSELoss()
    # criterion = nn.L1Loss()
    # criterion = nn.AbsCriterion()
    criterion = criterion.to(device)
    
    val_plot = []; train_plot = []
    val_loss_plot = []; train_loss_plot = []

    # Model Training
    for i_epoch in range(epochs):

        print("Epoch:", i_epoch, '/', epochs)

        val_loss = 0; val_acc = 0; train_loss = 0; train_acc = 0; train_loss2 = 0
        val_min = 0
        
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)

            yt_pred = model(X, ii)
            
            train_loss = criterion(yt_pred, y)

            train_loss2 += torch.sum(torch.pow(y - yt_pred, 2)).detach().cpu().numpy()

            train_acc += torch.sum(1 - torch.abs(y - yt_pred) / yt_pred).detach().cpu().numpy()
            
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

        model.eval()
        for X, y in validation_loader:
            with torch.no_grad():
                X, y = X.to(device), y.to(device)
                y_pred = model(X, ii)

                val_loss += torch.sum(torch.pow(y - y_pred, 2)).detach().cpu().numpy()

                val_acc += torch.sum(1 - torch.abs(y - y_pred) / y_pred).detach().cpu().numpy()

                val_min += torch.min(1 - torch.abs(y - y_pred)).detach().cpu().numpy()

        train_loss2 /= train_slice * data_size * out_piece
        train_acc /= train_slice * data_size * out_piece
        val_loss /= val_slice * data_size * out_piece
        val_acc /= val_slice * data_size * out_piece
        val_min /= val_slice * data_size * out_piece

        # val_plot.append(val_min)
        val_plot.append(val_acc)
        train_plot.append(train_acc)
        train_loss_plot.append(train_loss2)
        val_loss_plot.append(val_loss)

        # mean sum squared loss
        print( "Val Loss: " + str(round(val_loss, 4)) + ' Val Acc: ' + str(round(val_acc, 4)))


    #------------- Loss and Accuracy Plots -------------#
    if plot_curves == 1:

        val_plot = np.array(val_plot)
        train_plot = np.array(train_plot)
        val_plot[val_plot < 0] = 0
        train_plot[train_plot < 0] = 0
        val_plot[val_plot > 1] = 1
        train_plot[train_plot > 1] = 1

        fig, axs = plt.subplots(1, 2)
        del fig

        fontProperties = {'family':'serif',
            'weight' : 'normal', 'size' : 11}

        axs[0].plot(np.arange(epochs), val_loss_plot, color='crimson')
        axs[0].plot(np.arange(epochs), train_loss_plot, color='navy', linestyle='--')
        axs[1].plot(np.arange(epochs), val_plot, color='crimson')
        axs[1].plot(np.arange(epochs), train_plot, color='navy', linestyle='--')

        print('Validation loss:', val_loss_plot[-1])
        print('Train loss:', train_loss_plot[-1])
        print('Validation accuracy:', val_plot[-1])
        print('Train accuracy:', train_plot[-1])

        axs[0].tick_params(axis='x', direction='in')
        axs[0].tick_params(axis='y', direction='in')
        axs[0].set_aspect(aspect=1.0/axs[0].get_data_ratio())
        # axs[0].xlim(left = 0)
        axs[1].tick_params(axis='x', direction='in')
        axs[1].tick_params(axis='y', direction='in')
        axs[1].set_aspect(aspect=1.0/axs[1].get_data_ratio())
        # axs[1].xlim(left = 0)
        axs[0].set_xticklabels(axs[0].get_xticks().astype(int), fontProperties)
        axs[0].set_yticklabels(axs[0].get_yticks().astype(int), fontProperties)
        axs[1].set_xticklabels(axs[1].get_xticks().astype(int), fontProperties)
        axs[1].set_yticklabels(np.round(axs[1].get_yticks(), 2), fontProperties)

        plt.show()

    if parallel == False:
        print(round(ii/rows*100, 2), '%')

    return 0
