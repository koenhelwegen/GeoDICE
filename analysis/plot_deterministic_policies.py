import matplotlib.pyplot as plt
from handlecall import handle_call


def plot_deterministic_policy(args, datlst):

    # create axes
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)

    for _, optimal, _, timevec, _, descr, _ in datlst:

        # plot mitigation
        ax1.plot(timevec, [a[0] for a in optimal.actionvec], label=descr)

        # plot SRM
        ax2.plot(timevec, [a[1] for a in optimal.actionvec], label=descr)

    # layout mitigation
    plt.subplot(ax1)
    plt.xlabel("Time (years)")
    plt.ylabel("Mitigation (ratio of economic output)")
    plt.legend()

    # layout SRM
    plt.subplot(ax2)
    plt.xlabel("Time (years)")
    plt.ylabel("SRM (100 T S/yr)")
    plt.legend()

    plt.show()


if __name__ == "__main__":

    handle_call(plot_deterministic_policy, False)
