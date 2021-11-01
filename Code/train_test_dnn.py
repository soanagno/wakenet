from neuralWake import *
from optimisation import *
import synth_and_train as st

if synth == 1:
    """Model training main
    """

    # Start trainingtimer
    t0 = time.time()

    # Create the dataset
    X_train, X_val, X_test, y_train, y_val, y_test = st.create()
    # Set neural model
    model = wakeNet().to(device)

    if parallel == True and device == 'cpu':

        # Parallel CPU training
        print('Training on parallel CPU cores...')

        # Feed domain points to train the model
        lazy_results = []
        for pixel in range(rows):
            lazy_result = dask.delayed(st.training)(pixel, X_train, X_val, X_test,
                                                            y_train, y_val, y_test, model)
            lazy_results.append(lazy_result)

        results = dask.compute(*lazy_results, num_workers=para_workers)

    else:

        # Serial CPU tarining (or GPU training depending on set device)
        print('Training...')

        # Feed domain points to train the model
        for pixel in range(rows):
            st.training(pixel, X_train, X_val, X_test, y_train, y_val, y_test, model)

    # End training timer
    t1 = time.time()
    print('Training took: ', int(t1 - t0), ' seconds')

    # Save model weights
    torch.save(model, weights_path)

else:

    # Sets test case value
    test = int(input('Please enter the test case number (1-4): '))

    if test == 1:

        # Single and multiple wake comparisons
        # Recommeded denoise=7 (in info.json)

        # Single
        xs = np.array([1*D])
        ys = np.array([1*D])
        yws = [-30]
        compare(plots=True, yws=yws, ws=12, ti=0.12, xs=xs, ys=ys, print_times=True, single=False)

        # Multiple
        xs = np.array([1*D, 1*D, 1*D,
                       4.5*D, 4.5*D, 4.5*D,
                       8*D, 8*D, 8*D])
        ys = np.array([1*D, 3*D, 5*D,
                       2*D, 4*D, 6*D,
                       1*D, 3*D, 5*D])
        yws = [30, -30, 30, -30, 30, -30, 30, -30, 30, -30]

        # Multiple
        xs = np.array([1*D, 1*D,
                       8*D, 8*D])
        ys = np.array([1*D, 5*D,
                       1*D, 5*D])
        yws = [30, 30, -30, -30]
        compare(plots=True, yws=yws, ws=6, ti=0.12, xs=xs, ys=ys, print_times=True)

    if test == 2:

        # Superposition test
        # Recommeded denoise=7 (in info.json)
        
        # SOS 1
        ys = np.array([0, 0, 0, 4*D, 4*D, 4*D])
        xs = np.array([1*D, 8*D, 15*D, 1*D, 8*D, 15*D])
        yws = np.array([0, 0, 0, 0, 0, 0])
        compare(plots=True, yws=yws, ws=11, ti=0.12, xs=xs, ys=ys, print_times=True)

        # SOS 2
        ys = np.array([0, 1*D, 0.5*D])
        xs = np.array([1*D, 1*D, 4*D])
        yws = np.array([0, 0, 0])
        compare(plots=True, yws=yws, ws=11, ti=0.12, xs=xs, ys=ys, print_times=True)


    if test == 3:

        # Yaw Optimisation

        xs = np.array([1*D, 5.762*D])
        ys = np.array([1*D, 1*D])
        # compare(plots=True, yws=[18, 0], ws=9, ti=0.1, xs=xs, ys=ys, print_times=True)
        # exit()
        yawVsPowerContour(ws=9, ti=0.1, xs=xs, ys=ys, res=10, farm_opt=False)

        # Case A (yaw)
        xs = np.array([1*D, 1*D, 8*D, 8*D, 15*D, 15*D])
        ys = np.array([1*D, 7*D, 1*D, 7*D, 1*D, 7*D])

        florisOptimiser(ws=7, ti=0.11, layout_x=xs, layout_y=ys, plots=True)
        neuralOptimiser(ws=7, ti=0.11, xs=xs, ys=ys, plots=True, floris_gain=True)

        # Yaw power heatmaps
        heatmap(xs, ys, res=10, farm_opt=False)

        # Case B (yaw)
        xs = np.array([1*D, 1*D, 1*D, 4.5*D, 4.5*D,
                       8*D, 8*D, 8*D, 11.5*D, 11.5*D,
                       15*D, 15*D, 15*D, 18.5*D, 18.5*D])
        ys = np.array([1*D, 5*D, 9*D, 3*D, 7*D,
                       1*D, 5*D, 9*D, 3*D, 7*D,
                       1*D, 5*D, 9*D, 3*D, 7*D])

        florisOptimiser(ws=11, ti=0.11, layout_x=xs, layout_y=ys, plots=True)
        neuralOptimiser(ws=11, ti=0.11, xs=xs, ys=ys, plots=True, floris_gain=True)

        # Yaw power heatmaps
        heatmap(xs, ys, res=10, farm_opt=False)


    if test == 4:
        # Layout Optimisation

        # Case C
        xs = np.array([1*D, 1*D, 8*D, 8*D, 15*D, 15*D])
        ys = np.array([1*D, 5*D, 1*D, 5*D, 1*D, 5*D])

        florisOptimiser(ws=7, ti=0.05, layout_x=xs, layout_y=ys, plots=True, mode='farm')
        neuralOptimiser(ws=7, ti=0.05, xs=xs, ys=ys, plots=True, floris_gain=True, mode='farm')

        # Layout power heatmaps
        heatmap(xs, ys, farm_opt=True)
