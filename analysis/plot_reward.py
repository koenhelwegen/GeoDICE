import matplotlib.pyplot as plt
import numpy as np
from handlecall import handle_call


def main(args, datlst):

    # create axes
    ax1 = plt.subplot(131)
    ax2 = plt.subplot(132)
    ax3 = plt.subplot(133)

    dx = -1e-3

    for _ in range(args.repeat):
        for env, optimal, dpc, timevec, sample, descr, _ in datlst:
            # OPTIMAL #
            opt_reward = []
            env.reset()
            for a in optimal.actionvec:
                _, r, _, _ = env.step(a)
                opt_reward.append(r)

            # MITIGATION ONLY #
            mit_reward = []
            env.reset()
            for a in optimal.actionvec:
                _, r, _, _ = env.step(np.array([a[0], 0]))
                mit_reward.append(r)

            # BUSINESS AS USUAL #
            bua_reward = []
            env.reset()
            for a in optimal.actionvec:
                _, r, _, _ = env.step(np.array([0, 0]))
                bua_reward.append(r)

            # MARGINAL MITIGATION #
            ma0_reward = []
            env.reset()
            for a in optimal.actionvec:
                _, r, _, _ = env.step(a * [1 + dx, 1])
                ma0_reward.append(r)

            # MARGINAL SRM #
            ma1_reward = []
            env.reset()
            for a in optimal.actionvec:
                _, r, _, _ = env.step(a * [1, 1 + dx])
                ma1_reward.append(r)

            # plot mitigation
            ax1.plot(timevec, opt_reward, ".b", label="optimal ({})".format(descr))
            ax1.plot(timevec, bua_reward, ".r", label="BAU ({})".format(descr))
            ax1.plot(
                timevec, opt_reward, ".g", label="mitigation only ({})".format(descr)
            )

            ax2.plot(
                timevec,
                (np.array(opt_reward) - np.array(bua_reward)) / dx,
                ".r",
                label="wrt. BAU ({})".format(descr),
            )
            ax2.plot(
                timevec,
                (np.array(opt_reward) - np.array(mit_reward)) / dx,
                ".g",
                label="wrt. mitigation only  ({})".format(descr),
            )

            ax3.plot(
                timevec,
                (np.array(opt_reward) - np.array(ma0_reward)) / dx,
                ".b",
                label="Mitigation ({})".format(descr),
            )
            ax3.plot(
                timevec,
                (np.array(opt_reward) - np.array(ma1_reward)) / dx,
                ".r",
                label="SRM ({})".format(descr),
            )

    plt.subplot(ax1)
    plt.xlabel("Time (years)")
    plt.ylabel("Reward (step by step)")
    plt.legend()

    plt.subplot(ax2)
    plt.xlabel("Time (years)")
    plt.ylabel("Relative Reward (step by step)")
    plt.legend()

    plt.subplot(ax3)
    plt.xlabel("Time (years)")
    plt.ylabel("10\% difference")
    plt.legend()

    plt.show()


if __name__ == "__main__":
    handle_call(main)
