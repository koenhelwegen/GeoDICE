from handlecall import handle_call
import numpy as np


def print_statistics(args, datlst):

    for env, optimal, dpc, timevec, samplevec, descr, longdescr in datlst:

        sample = samplevec[0]
        experiment = env.get_experiment()
        nodes = dpc.get_n_approx_nodes()[0]

        print("\n")
        print(longdescr)
        print("-----")

        # RELATIVE PERFORMANCE #

        # get baseline performance
        base_utility = 0
        env.reset()
        for _ in timevec:
            _, r, _, _ = env.step(np.array([0, 0]))
        print(
            "Sample Utility: {} ({})".format(
                env.convert_utility(sample.utility), sample.utility
            )
        )
        print(
            "Expected utility at 2005: {} ({})".format(
                env.convert_utility(dpc.valuefun(0, optimal.trajectory[0], 0)),
                dpc.valuefun(0, optimal.trajectory[0], 0),
            )
        )
        relper = (sample.utility - base_utility) / (optimal.utility - base_utility)
        print("Relative performance: {}".format(relper))

        if args.savefile is not None:
            with open(args.savefile, "a+") as f:
                f.write(
                    "% Relative performance, exp {} with {} nodes: {}\n".format(
                        experiment, nodes, relper
                    )
                )
                f.write(
                    "\\num\\def\\relper{}{}{{{:.6f}\\%}}\n\n".format(
                        nodes, experiment, relper * 100
                    )
                )

        timevec = np.array(timevec)

        # POLICY DIFFERENCE #
        x = np.array(sample.actionvec[:31]).reshape(-1, 1)
        y = np.array(optimal.actionvec[:31]).reshape(-1, 1)
        x[y == 0] = 1
        y[y == 0] = 1
        max_policy_dif = max(abs((x - y) / y))[0]
        print("Largest policy difference: {}%".format(max_policy_dif * 100))
        if args.verbose == -1:
            for t in range(31):
                print(
                    "Optimal: ({}, {}) | DP: ({}, {})".format(
                        optimal.actionvec[t][0],
                        optimal.actionvec[t][1],
                        sample.actionvec[t][0],
                        sample.actionvec[t][1],
                    )
                )
        if args.savefile is not None:
            with open(args.savefile, "a+") as f:
                f.write(
                    "% Max. policy diff., exp {} with {} nodes: {}\n".format(
                        experiment, nodes, max_policy_dif
                    )
                )
                f.write(
                    "\\num\\def\\maxpoldif{}{}{{{:.2f}\\%}}\n\n".format(
                        nodes, experiment, max_policy_dif * 100
                    )
                )

        # TRAJECTORY DIFFERENCE #
        if env.get_modeltype() in ["lrgeodice", "lrgd2"]:
            dp_co2 = np.array([sum(st[1:5]) for st in sample.trajectory[:31]])
            dp_t = np.array([sum(st[5:7]) for st in sample.trajectory[:31]])
            opt_co2 = np.array([sum(st[1:5]) for st in optimal.trajectory[:31]])
            opt_t = np.array([sum(st[5:7]) for st in optimal.trajectory[:31]])

            max_traj_dif = max(
                max(abs((dp_co2 - opt_co2) / opt_co2)), max(abs((dp_t - opt_t) / opt_t))
            )
            print("Largest trajectory difference: {}%".format(max_traj_dif * 100))

            if args.savefile is not None:
                with open(args.savefile, "a+") as f:
                    f.write(
                        "% Max. traj diff., exp {} with {} nodes: {}\n".format(
                            experiment, nodes, max_traj_dif
                        )
                    )
                    f.write(
                        "\\num\\def\\maxtrajdif{}{}{{{:.2f}\\%}}\n\n".format(
                            nodes, experiment, max_traj_dif * 100
                        )
                    )

        # PREDICTION ERROR #
        v_pred = env.convert_utility(sample.predictionvec[0])
        v_true = env.convert_utility(sample.utility)
        print("Prediction error: {}%".format(100 * v_pred / v_true - 100))


if __name__ == "__main__":
    handle_call(print_statistics)
