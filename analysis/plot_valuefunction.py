# standard imports
import matplotlib.pyplot as plt
import numpy as np
import copy
import math
import lib.approx.chebyshev as cheb
import pickle
from handlecall import handle_call

# project imports
import output.settings as settings

varname = dict()
varname[0] = "Capital"
varname[1] = "Permanent CO2"
varname[2] = "Slow CO2"
varname[3] = "Medium CO2"
varname[4] = "Fast CO2"
varname[5] = "Fast temperature"
varname[6] = "Slow temperature"

PLOT_CENTERSTATE = False


def main(args, datlst):
    """ Plot value function around a certain point in state space.

    Note that the true estimate is based on the fitted valuefunction in the
    next step, and therefore should not be confused with the true valuefunction.

    Specific state is selected using -c, with options:
    -3: Tipping point
    -2: SRM discontinuity
    -1: state with low co2 & temperature
    0: random state
    1: optimal state
    2: state that is overvalued in fit
    3: state where optimal SRM policy is negative
    4: state that is undervalued in fit
    """

    # general settings
    center = args.center
    verbose = args.verbose
    t = args.time
    print_result = args.print_result

    for env, optimal, dpc, timevec, sample, descr, _ in datlst:

        experiment = env.get_experiment()

        # SETTINGS #
        SOTW_ = args.sotw[0]
        # bnds_low, bnds_high = modeltools.load_domain(env)
        bnds_low, bnds_high = dpc.get_domain()
        if args.dimension[0] == -1:
            dimvec = list(range(env.get_state_dimension()))
        else:
            dimvec = args.dimension
        print(
            "Plotting cross-section valuefunction at time {} in dimensions {}, sotw {}".format(
                t, dimvec, args.sotw
            )
        )

        # SELECT STATE #
        # select state around which to plot valuefunction
        randomstate = cheb.box_to_domain(
            np.random.rand(env.get_state_dimension()) * 2 - 1, bnds_low[t], bnds_high[t]
        )

        if center == -2:
            randomstate = np.array([240, 382, 45, 60, 1, -0.2, -0.2])
            randomstate = np.array([160, 300, 0, 0, 0, -0.5, -1.2])

        if center == -3:
            randomstate = np.array([265, 347, 63, 36, 6, 0.21, 0.25])

        if center == 2:
            counter = 0
            while dpc.optimize_action_value(
                t, randomstate, SOTW_, np.array([0.3, 0.15])
            )[0] > dpc.valuefun(t, randomstate, SOTW_):
                randomstate = cheb.box_to_domain(
                    np.random.rand(env.get_state_dimension()) * 2 - 1,
                    bnds_low[t],
                    bnds_high[t],
                )
                counter += 1
            print("Found abnormal state after {} trials".format(counter))

        if center == 4:
            counter = 0
            while dpc.optimize_action_value(
                t, randomstate, SOTW_, np.array([0.3, 0.15])
            )[0] < dpc.valuefun(t, randomstate, SOTW_):
                randomstate = cheb.box_to_domain(
                    np.random.rand(env.get_state_dimension()) * 2 - 1,
                    bnds_low[t],
                    bnds_high[t],
                )
                counter += 1
            print("Found abnormal state after {} trials".format(counter))

        if center == 3:
            counter = 0
            print("Looking for state with negative SRM")
            while (
                dpc.optimize_action_value(t, randomstate, SOTW_, np.array([0.3, 0.15]))[
                    1
                ][1]
                >= 0
            ):
                randomstate = cheb.box_to_domain(
                    np.random.rand(env.get_state_dimension()) * 2 - 1,
                    bnds_low[t],
                    bnds_high[t],
                )
                counter += 1
            print("Found abnormal state after {} trials".format(counter))

        if center == -1:
            while sum(randomstate[5:7]) > -0.5 or sum(randomstate[1:5]) > 278:
                randomstate = cheb.box_to_domain(
                    np.random.rand(env.get_state_dimension()) * 2 - 1,
                    bnds_low[t],
                    bnds_high[t],
                )
            print(
                "Found state with negative temperature and low carbon.\nT_AT: {} C, M_AT: {} ppm".format(
                    sum(randomstate[5:7]), sum(randomstate[1:5])
                )
            )

        # PLOTTING #
        for dim in dimvec:

            # select plot
            if len(dimvec) > 1:
                plt.subplot(math.ceil(len(dimvec) / 2.0 + 1), 2, 1 + dim)

            # make copy of our center state
            if center != 1:
                centerstate = randomstate
            else:
                centerstate = optimal.trajectory[t]
            st = copy.deepcopy(centerstate)

            if len(dimvec) == 1:
                print(st)

            if args.sotw == [2] or args.sotw == [3]:
                for SOTW in [0]:
                    xvec = np.linspace(bnds_low[t][dim], bnds_high[t][dim], 100)
                    yvec = []
                    for x in xvec:
                        st[dim] = x
                        yvec.append(dpc.valuefun(t, st, SOTW))
                    plt.plot(xvec, yvec, "-", linewidth=3, c="grey")
                    plt.xlim([min(xvec), max(xvec)])

            # PLOT VALUEFUNCTION FIT #
            for SOTW in args.sotw:
                xvec = np.linspace(bnds_low[t][dim], bnds_high[t][dim], 100)
                yvec = []
                for x in xvec:
                    st[dim] = x
                    yvec.append(dpc.valuefun(t, st, SOTW))
                plt.plot(xvec, yvec, label="Fitted".format(SOTW))
                plt.xlim([min(xvec), max(xvec)])

            # PLOT CENTER STATE #
            # for the center state, show both the fitted and the true estimate
            if PLOT_CENTERSTATE:
                st = copy.deepcopy(centerstate)
                if center == 1:
                    plt.scatter(
                        st[dim], dpc.valuefun(t, st, SOTW_), c="r", label="Fit opt. st."
                    )
                else:
                    plt.scatter(
                        st[dim],
                        dpc.valuefun(t, st, SOTW_),
                        c="orange",
                        label="Fit rand. st.",
                    )

                y = dpc.optimize_action_value(t, st, SOTW_, np.array([0.3, 0.15]))
                if center == 1:
                    plt.scatter(
                        st[dim], y[0], marker="*", c="r", label="True estimate opt. st."
                    )
                else:
                    plt.scatter(
                        st[dim],
                        y[0],
                        marker="*",
                        c="orange",
                        label="True estimate rand. st.",
                    )

            # PLOT CHEBYSHEV NODES #
            m = dpc.get_n_approx_nodes()
            if type(m) == list:
                m = m[dim]
            for j in range(m):
                x = -math.cos((2 * j + 1) * math.pi / (2 * m))
                st[dim] = (
                    bnds_low[t][dim]
                    + (bnds_high[t][dim] - bnds_low[t][dim]) * (x + 1) / 2.0
                )
                y = dpc.optimize_action_value(t, st, SOTW_, np.array([0.3, 0.05]))
                env.setstate(t, st, SOTW_)
                if abs(sum(env.scenario_probabilities()) - 1) > 1e-8:
                    print(env.scenario_probabilities())
                plt.scatter(st[dim], y[0], c="g")
            plt.scatter([], [], c="g", label="Approx. nodes")

            # PLOT TRUE ESTIMATE #
            xvec = np.linspace(bnds_low[t][dim], bnds_high[t][dim], 40)
            for SOTW in args.sotw:
                yvec = []
                avec = []
                for x in xvec:
                    st[dim] = x
                    y = dpc.optimize_action_value(t, st, SOTW, np.array([0.3, 0.15]))
                    yvec.append(y[0])
                    avec.append(y[1])
                plt.plot(xvec, yvec, "--", label="True".format(SOTW))
            SOTW = 0

            # store valuefunction for later analysis
            # if dim == 6:
            #     with open('data/other/valuefun.pickle', 'wb') as f:
            #         pickle.dump((xvec, yvec), f)

            # some quick layout
            plt.xlabel("Dimension {}".format(dim))
            plt.ylabel("Value")

            # add legend if needed
            if dim == 0 and len(args.sotw) > 1 and len(dimvec) > 1:
                plt.legend()

            if dim == 6:
                if len(dimvec) > 1:
                    plt.subplot(4, 2, 8)
                    plt.plot(xvec, [a[1] for a in avec])
                else:
                    with open(
                        "output/data/valuefunactexp{}.pickle".format(experiment), "wb"
                    ) as f:
                        pickle.dump([xvec, avec], f)

        fig = plt.gcf()

        # LAYOUT (for single dimension plot) #
        if len(dimvec) == 1:
            ax = plt.gca()

            ax.tick_params(labelsize=settings.tick_fontsize)
            plt.xlabel(
                varname[dimvec[0]],
                size=settings.label_fontsize,
                labelpad=settings.label_padding,
            )
            plt.ylabel(
                "Value", size=settings.label_fontsize, labelpad=settings.label_padding
            )

            if not args.suppress_legend:
                plt.legend(fontsize=settings.legend_fontsize)

            plt.ylim([2008, 2018])
            fig.tight_layout()

            if args.savefile is not None:
                settings.savefig(args.savefile)

        plt.show()


if __name__ == "__main__":
    handle_call(main)
