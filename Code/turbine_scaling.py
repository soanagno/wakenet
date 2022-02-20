from neuralWake import *
from superposition import *
from synth_and_train import *
from optimisation import *
import synth_and_train as dat


if synth == 1:

    # Plot wake dataset sample
    dat.Create(plots=True)

else:

    # ------------ Computational time vs Superimposed turbines scaling ------------ #

    xs = np.array(
        [
            0,
            0,
            0,
            7 * D,
            7 * D,
            7 * D,
            14 * D,
            14 * D,
            14 * D,
            21 * D,
            21 * D,
            21 * D,
            28 * D,
            28 * D,
            28 * D,
            35 * D,
            35 * D,
            35 * D,
            42 * D,
            42 * D,
            42 * D,
            49 * D,
            49 * D,
            49 * D,
            56 * D,
        ]
    )
    ys = np.array(
        [
            0 * D,
            2 * D,
            4 * D,
            1 * D,
            3 * D,
            5 * D,
            0 * D,
            2 * D,
            4 * D,
            1 * D,
            3 * D,
            5 * D,
            0 * D,
            2 * D,
            4 * D,
            1 * D,
            3 * D,
            5 * D,
            0 * D,
            2 * D,
            4 * D,
            1 * D,
            3 * D,
            5 * D,
            0 * D,
        ]
    )
    yws = np.zeros(xs.size)

    iterations = 5
    max_turbines = 20

    floris_time_plot = np.zeros(max_turbines)
    neural_time_plot = np.zeros(max_turbines)

    for i in range(max_turbines):

        print("No. of turbines:", i)

        for _ in range(iterations):
            floris_time, neural_time = compare(
                yws=yws[: i + 1],
                ws=7,
                ti=0.05,
                xs=xs[: i + 1],
                ys=ys[: i + 1],
                print_times=False,
                timings=True,
            )

            floris_time_plot[i] += floris_time
            neural_time_plot[i] += neural_time

        floris_time_plot[i] /= iterations
        neural_time_plot[i] /= iterations

    fig, ax = plt.subplots(1)

    # plt.plot(np.arange(1, max_turbines+1), floris_time_plot/100, color='navy', linestyle='--')
    plt.plot(
        np.arange(1, max_turbines + 1), floris_time_plot, color="navy", linestyle="--"
    )
    plt.plot(np.arange(1, max_turbines + 1), neural_time_plot, color="crimson")
    plt.xscale("log")
    plt.yscale("log")

    fontProperties = {"family": "serif", "weight": "normal", "size": 11}

    plt.tick_params(axis="x", direction="in")
    plt.tick_params(axis="y", direction="in")
    # plt.set_aspect(aspect=1.0/plt.get_data_ratio())

    ax.set_xticklabels(ax.get_xticks().astype(int), fontProperties)
    ax.set_yticklabels(ax.get_yticks(), fontProperties)

    plt.show()
