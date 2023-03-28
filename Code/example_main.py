from neuralWake import *
from optimisation import *
import synth_and_train as st


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

    return round(floris_power_0/1e6, 2)

def main():


    if train_net == True:

        # Start training timer
        t0 = time.time()
        # Create the dataset
        X_train, X_val, X_test, y_train, y_val, y_test = st.create()
        # Set neural model
        model = wakeNet().to(device)
        # Feed domain points to train the model
        print("Training...")
        vloss_plot, tloss_plot, v_plot, t_plot = \
        st.training(X_train, X_val, X_test, y_train, y_val, y_test, model, plot_curves=1, saveas='tcurv')
        # End training timer
        t1 = time.time()
        print("Training took: ", int(t1 - t0), " seconds")

    else:

        # Set neural model
        model = wakeNet().to(device)
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.eval().to(device)

        # Sets test case value
        test = int(input("Please enter the test case number (1-4): "))

        if test == 1:

            # Single and multiple wake comparisons

            # Single
            xs = np.array([D])
            ys = np.array([D])
            yws = [-30]
            compare(
                plots=True,
                yws=yws,
                ws=11,
                ti=0.05,
                xs=xs,
                ys=ys,
                print_times=True,
                single=False,
            )

            # Multiple 1
            xs = np.array([1*D, 1*D, 1*D,
                        4.5*D, 4.5*D, 4.5*D,
                        8*D, 8*D, 8*D])
            ys = np.array([1*D, 3*D, 5*D,
                        2*D, 4*D, 6*D,
                        1*D, 3*D, 5*D])
            yws = [30, -30, 30, -30, 30, -30, 30, -30, 30, -30]

            compare(
                plots=True,
                yws=yws,
                ws=11,
                ti=0.05,
                xs=xs,
                ys=ys,
                print_times=True,
                single=False,
            )

        if test == 2:


            # Case A (yaw) M3
            xs = np.array([1*D, 1*D, 8*D, 8*D, 15*D, 15*D])
            ys = np.array([1*D, 7*D, 1*D, 7*D, 1*D, 7*D])
            florisOptimiser(ws=11, ti=0.05, layout_x=xs, layout_y=ys, plots=True)
            neuralOptimiser(ws=11, ti=0.05, xs=xs, ys=ys, plots=True, floris_gain=True)

            # Yaw power heatmaps
            heatmap(xs, ys, res=3, farm_opt=False)

            # Case B (yaw) M2
            xs = np.array([1*D, 1*D, 1*D, 4.5*D, 4.5*D,
                        8*D, 8*D, 8*D, 11.5*D, 11.5*D,
                        15*D, 15*D, 15*D, 18.5*D, 18.5*D])
            ys = np.array([1*D, 5*D, 9*D, 3*D, 7*D,
                        1*D, 5*D, 9*D, 3*D, 7*D,
                        1*D, 5*D, 9*D, 3*D, 7*D])
            florisOptimiser(ws=11, ti=0.05, layout_x=xs, layout_y=ys, plots=False)
            neuralOptimiser(ws=11, ti=0.05, xs=xs, ys=ys, plots=False, floris_gain=True)

            # Yaw power heatmaps
            heatmap(xs, ys, res=3, farm_opt=False)

        if test == 3:

            # Case C (Layout)

            # 6-turb
            xs = np.array([1*D, 1*D, 8*D, 8*D, 15*D, 15*D])
            ys = np.array([1*D, 5*D, 1*D, 5*D, 1*D, 5*D])

            # # 15-turb
            # xs = np.array([1*D, 1*D, 1*D, 4.5*D, 4.5*D,
            #     8*D, 8*D, 8*D, 11.5*D, 11.5*D,
            #     15*D, 15*D, 15*D, 18.5*D, 18.5*D])
            # ys = np.array([1*D, 5*D, 9*D, 3*D, 7*D,
            #     1*D, 5*D, 9*D, 3*D, 7*D,
            #     1*D, 5*D, 9*D, 3*D, 7*D])

            neuralOptimiser(ws=11.0, ti=0.05, xs=xs, ys=ys, plots=True, floris_gain=True, mode='farm')
            florisOptimiser(ws=11.0, ti=0.05, layout_x=xs, layout_y=ys, plots=True, mode='farm')

            # Layout power heatmaps
            heatmap(xs, ys, res=10, farm_opt=True)
            heatmap(xs, ys, res=3, farm_opt=True)


if __name__=="__main__":
    main()
