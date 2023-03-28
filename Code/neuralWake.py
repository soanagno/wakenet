from packages import *
from initialisations import *


class wakeNet(nn.Module):
    """
    wakeNet class definition
    """

    def __init__(self, inputs=3, hidden_neurons=[100, 200]):
        """
        wakeNet initializations

        Args:
            u_stream (torch float array) Inputs of training step.
            ws (float) Wind speed.
            ti (float) Turbulence intensity.
            yw (float) Yaw angle.
            hb (float) Hub height.
            model (torch model) Passes the neural model to be used.
            timings (boolean, optional) Prints and output timings of both Neural
                and Analytical calculations.

        Returns:
            gauss_time, neural_time, error (floats) Analytical, Neural timings and
                absolute mean error between the Analytical and Neural wake deficits.

            or

            final (2D numpy float array) Wake profile with u_stream background velocity.
        """

        super(wakeNet, self).__init__()


        # Parameters
        self.inputSize = inputs
        self.outputSize = out_piece
        self.hidden_neurons = hidden_neurons
        self.layers = len(self.hidden_neurons) + 1
        iSize = [self.inputSize] + self.hidden_neurons + [self.outputSize]

        # Initialisation of linear layers
        self.fc = []
        # Append layers
        for psi in range(self.layers):
            self.fc.append(nn.Linear(iSize[psi], iSize[psi+1], bias=True).to(device))
        self.fc1 = nn.Linear(iSize[0], iSize[1], bias=True).to(device)
        self.fc2 = nn.Linear(iSize[1], iSize[2], bias=True).to(device)
        self.fc3 = nn.Linear(iSize[2], iSize[3], bias=True).to(device)
        # Initialisation of batchnorm layers
        self.fcb = []
        # Append layers
        for psi in range(self.layers-1):
            self.fcb.append(nn.BatchNorm1d(iSize[psi+1], affine=False).to(device))
        self.fcb1 = nn.BatchNorm1d(iSize[1], affine=False).to(device)
        self.fcb2 = nn.BatchNorm1d(iSize[2], affine=False).to(device)

        # Dropout
        self.drop = nn.Dropout(0.2).to(device)  # 20% probability

        # Activation functions
        self.act = nn.Tanh().to(device)
        self.act2 = self.purelin

    def tansig(self, s):
        return 2 / (1 + torch.exp(-2 * s)) - 1

    def purelin(self, s):
        return s

    @staticmethod
    def tiVsVel(n, weather=weather, plots=False):
        """Make ti vs speeds distribution"""

        if plots == True:
            np.random.seed(89)
            xs0 = (np.random.rand(data_size) * (ws_range[1] - ws_range[0]) + ws_range[0])  # ws
            np.random.seed(42)
            ys0 = (np.random.rand(data_size) * (ti_range[1] - ti_range[0]) + ti_range[0])  # ti

            lower, upper = ws_range[0], ws_range[1]
            s = 1e-9
            mu, sigma = 3, 8
            xx = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
            xs = xx.rvs(n)
            yy = 2 ** (1 / (xs + s) / 6) - 0.9

            rs = []
            for _ in range(n):
                rs.append(-0.01 + random.random() * 0.02)
            ys = 2 ** (1 / (xs + s) / 6) - 0.9 + rs * (1 + 60 * (yy - 0.1))

            plt.scatter(xs, ys, s=0.5)
            plt.show()
            exit()

        if weather == False:
            np.random.seed(89)
            xs = (np.random.rand(data_size) * (ws_range[1] - ws_range[0]) + ws_range[0])  # ws
            np.random.seed(42)
            ys = (np.random.rand(data_size) * (ti_range[1] - ti_range[0]) + ti_range[0])  # ti

        else:
            lower, upper = ws_range[0], ws_range[1]
            s = 1e-9
            mu, sigma = 3, 8
            xx = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
            xs = xx.rvs(n)
            yy = 2 ** (1 / (xs + s) / 6) - 0.9

            rs = []
            for _ in range(n):
                rs.append(-0.01 + random.random() * 0.02)
            ys = 2 ** (1 / (xs + s) / 6) - 0.9 + rs * (1 + 60 * (yy - 0.1))

        return xs, ys

    def forward(self, X):
        """
        Performs a forward step during training.

        Args:
            X (torch float array) Inputs of training step.
            point (int): Number of individual sub-network to be trained.

        Returns:
            out (torch float array) Turbine wake output.
        """

        X = X.to(device)

        if train_net == 0:
            X = X.view(1, -1)
        X = self.fc1(X)
        X = self.act(X)
        X = self.fcb1(X)

        if train_net == 0:
            X = X.view(1, -1)
        X = self.fc2(X)
        X = self.act(X)
        X = self.fcb2(X)
        out = self.act2(self.fc3(X))

        return out

    def saveWeights(self, model):
        torch.save(model, "NN")

    def compareContour(
        self,
        u_stream,
        ws,
        ti_ar,
        yw,
        hb,
        model,
        result_plots=result_plots,
        timings=False,
    ):
        """
        Performs a forward step during training.

        Args:
            u_stream (torch float array) Inputs of training step.
            ws (float) Wind speed.
            ti (float) Turbulence intensity.
            yw (float) Yaw angle.
            hb (float) Hub height.
            model (torch model) Passes the neural model to be used.
            timings (boolean, optional) Prints and output timings of both Neural
                and Analytical calculations.

        Returns:
            gauss_time, neural_time, error (floats) Analytical, Neural timings and
                absolute mean error between the Analytical and Neural wake deficits.

            or

            final (2D numpy float array) Wake profile with u_stream background velocity.
        """

        tmp = yw
        yw = np.zeros(1)
        yw[0] = tmp
        hb = np.array(hb)

        ti = ti_ar[0]
        if timings == True or result_plots == True:

            t0 = time.time()

            # Set Floris parameters
            if curl == True:
                fi.floris.farm.set_wake_model("curl")  # curl model
            if inputs == 1:
                fi.reinitialize_flow_field(wind_speed=ws)
                fi.calculate_wake()
            if inputs == 2:
                fi.reinitialize_flow_field(wind_speed=ws)
                fi.reinitialize_flow_field(turbulence_intensity=ti)
                fi.calculate_wake()
            if inputs == 3:
                fi.reinitialize_flow_field(wind_speed=ws)
                fi.reinitialize_flow_field(turbulence_intensity=ti)
                fi.calculate_wake(yaw_angles=yw)

            if inputs == 4:
                fi.reinitialize_flow_field(wind_speed=ws)
                fi.reinitialize_flow_field(turbulence_intensity=ti)
                fi.change_turbine([0], {"yaw_angle": yw})
                cut_plane = fi.get_hor_plane(
                    height=hb,
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
                dimy, dimx
            )
            t1 = time.time()
            # Get analytical model timing
            gauss_time = t1 - t0
            # Keep min value for plotting
            vmin = np.min(u_mesh)
            vmax = np.max(u_mesh)

        # Initialise model for evaluation
        model.eval().to(device)
        t0 = time.time()
        # Initialise neural output vector
        neural = np.zeros(dimx * dimy)
        ti = ti_ar[1]

        # Normalisation
        speed_norm = ((ws - ws_range[0]) / (ws_range[1] - ws_range[0]) - 0.5) * 3
        ti_norm = ((ti - ti_range[0]) / (ti_range[1] - ti_range[0]) - 0.5) * 3
        yw_norm = ((yw - yw_range[0]) / (yw_range[1] - yw_range[0]) - 0.5) * 3
        hbs_norm = ((hb - hb_range[0]) / (hb_range[1] - hb_range[0]) - 0.5) * 3

        # Make input tensor
        if inputs == 1:
            inpt = torch.tensor(([speed_norm]), dtype=torch.float)
        elif inputs == 2:
            inpt = torch.tensor(([speed_norm, ti_norm]), dtype=torch.float)
        elif inputs == 3:
            inpt = torch.tensor(([speed_norm, ti_norm, yw_norm]), dtype=torch.float)
        elif inputs == 4:
            inpt = torch.tensor(
                ([speed_norm, ti_norm, yw_norm, hbs_norm]), dtype=torch.float
            )

        model.eval().to(device)
        neural = model(inpt).detach().cpu().numpy()

        # Apply Filter to replace backround with u_stream (helps with scattering)
        if fltr < 1.0:
            neural[neural > ws * fltr] = ws

        if cubes == 1:
            # Compose 2D velocity deficit made of blocks

            dd = dim1 * dim2
            jj = 0
            ii = 0
            alpha = np.zeros((dimy, dimx))
            for k in range(int(dimx * dimy / (dim1 * dim2))):

                alpha[ii : ii + dim1, jj : jj + dim2] = np.reshape(
                    neural[k * dd : k * dd + dd], (dim1, dim2)
                )

                jj += dim2
                if jj >= dimx:
                    jj = 0
                    ii += dim1

            neural = alpha.T

        else:
            if row_major == 0:
                # Compose 2D velocity deficit column-wise
                neural = np.reshape(neural, (dimx, dimy)).T
            else:
                # Compose 2D velocity deficit row-wise
                neural = np.reshape(neural, (dimy, dimx))

        t1 = time.time()
        # Get neural timing
        neural_time = t1 - t0

        # ----------------- Plot wake deficit results -----------------#
        if timings == True or result_plots == True:

            if result_plots == True:
                cmap = "coolwarm"

                fig, axs = plt.subplots(2)
                fig.suptitle("Velocities(m/s): Analytical (top), Neural (bot)")
                im1 = axs[0].imshow(
                    u_mesh,
                    vmin=vmin,
                    vmax=vmax,
                    cmap=cmap,
                    extent=[x_bounds[0], x_bounds[1], y_bounds[0], y_bounds[1]],
                )

                fig.colorbar(im1, ax=axs[0])
                im2 = axs[1].imshow(
                    neural,
                    vmin=vmin,
                    vmax=vmax,
                    interpolation=None,
                    cmap=cmap,
                    extent=[x_bounds[0], x_bounds[1], y_bounds[0], y_bounds[1]],
                )

                fig.colorbar(im2, ax=axs[1])

                plt.show()

            max_val = np.max(u_mesh)

            if timings == True:
                absdifsum = np.sum(np.abs(u_mesh - neural))
                error = round(1 / (dimx * dimy) * absdifsum / max_val * 100, 2)

                if result_plots == True:
                    print("Abs mean error (%): ", error)

            if result_plots == True:
                plt.imshow(
                    (np.abs(u_mesh - neural) / max_val * 100),
                    vmax=20,
                    extent=[x_bounds[0], x_bounds[1], y_bounds[0], y_bounds[1]],
                    cmap=cmap,
                )
                plt.colorbar()
                plt.title("Abs difference")
                plt.show()

                # ----- Y-Transect plots ----- #
                dx = 6.048
                tlist = np.array([3*D/dx, 6.5*D/dx, 10*D/dx]).astype(int)
                transects = tlist.size  # defines the number of transects
                fig, axs = plt.subplots(1, transects, sharey=False)
                cnt = 0
                for indx in tlist:

                    yy1 = u_mesh[:, indx]  # FLORIS transect
                    yy2 = neural[:, indx]  # CNN transect
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
                plt.show()

        final = np.copy(neural)
        # Replace current turbine inlet speed (ws) with farm u_stream (for superimposed wakes)
        final[final == ws] = u_stream

        if timings == True:
            return gauss_time, neural_time, error
        else:
            return final
