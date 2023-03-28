from neuralWake import *
from superposition import *
from synth_and_train import *
from optimisation import *
import synth_and_train as dat


if train_net == 1:

    # Plot wake dataset sample
    dat.Create(plots=True)

else:

    # ------------ Computational time vs Superimposed turbines scaling ------------ #

    iterations = 3
    mm = 4
    max_turbines = 6*mm
    saveas = "scaling"+str(max_turbines)+" "+device

    xs = [
            0,
            0,
            0,
            7 * D,
            7 * D,
            7 * D
         ]

    ys = [
            0 * D,
            2 * D,
            4 * D,
            1 * D,
            3 * D,
            5 * D
         ]

    cnt = 2
    for i in range(int(max_turbines/6+.5)-1):
        xs += [7*cnt*D, 7*cnt*D, 7*cnt*D] + [7*(cnt+1)*D, 7*(cnt+1)*D, 7*(cnt+1)*D]
        ys += ys[:6]
        cnt+=1

    xs = np.array(xs)
    ys = np.array(ys)
    yws = np.zeros(xs.size)

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
        np.arange(1, max_turbines + 1), floris_time_plot, color="navy", linestyle="--", label='FLORIS'
    )
    plt.plot(
        np.arange(1, max_turbines + 1), neural_time_plot, color="crimson", label='wakeNet'
    )
    plt.xscale("log")
    plt.yscale("log")

    plt.tick_params(axis="x", direction="in")
    plt.tick_params(axis="y", direction="in")

    plt.legend()
    plt.show()
    fig.savefig("figures/"+str(saveas), dpi=1200)
