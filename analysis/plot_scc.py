# standard imports
import matplotlib.pyplot as plt

# project imports
import lib.models.tools as modeltools
from handlecall import handle_call


def plot_scc(args, datlst):

    for env, optimal, dpc, timevec, sample, descr, _ in datlst:

        dx = 1e-0
        timevec = list(filter(lambda t: t < 2305, timevec))
        experiment = env.get_experiment()
        nodes = dpc.get_n_approx_nodes()[0]

        sample = dpc.sample_forward_trajectory()
        env_nonoise = modeltools.load_model(
            modeltype=env.get_modeltype(),
            noise=False,
            stepsize=env.get_stepsize(),
            experiment=env.get_experiment(),
        )
        # calculate DP & optimal SCC along optimal trajectory
        dp_scc = []
        opt_scc = []
        for t in range(len(timevec)):
            v = t < args.verbose
            dp_scc.append(
                dpc.estimate_SCC(t, sample.trajectory[t], sample.sotwvec[t], dx, v)
            )
            # if not args.noise:
            opt_scc.append(modeltools.scc(env_nonoise, t, verbose=v, dx=dx))
            # else:
            #     opt_scc.append(1)
            if (t < args.print_result or args.print_result == -1) and t >= args.verbose:
                print(
                    "Year: {} | SOTW: {}| DP SCC: {:.2f} | Optimal SCC: {:.2f} | Ratio: {}".format(
                        timevec[t],
                        sample.sotwvec[t],
                        dp_scc[-1],
                        opt_scc[-1],
                        dp_scc[-1] / opt_scc[-1],
                    )
                )

            if args.savefile is not None and timevec[t] in [2015, 2205]:
                with open(args.savefile, "a+") as f:
                    f.write(
                        "% {} SCC difference experiment {}, nodes {}: {}%\n".format(
                            timevec[t],
                            experiment,
                            nodes,
                            (dp_scc[t] / opt_scc[t] - 1) * 100,
                        )
                    )
                    f.write(
                        "\\num\\def\\sccdif{}{}{}{{{:.1f}\\%}}%\n\n".format(
                            nodes,
                            timevec[t],
                            experiment,
                            (dp_scc[t] / opt_scc[t] - 1) * 100,
                        )
                    )

        plt.plot(timevec, dp_scc, "-", label="DP SCC ({}y)".format(env.get_stepsize()))

        # if not args.noise:
        plt.plot(
            timevec, opt_scc, "--", label="NLO SCC ({}y)".format(env.get_stepsize())
        )

    # plt.xlim([2005, 2305])
    plt.xlabel("Time (years)")
    plt.ylabel("SCC")
    plt.legend()

    if args.savefile is None:
        plt.show()


if __name__ == "__main__":
    handle_call(plot_scc)
