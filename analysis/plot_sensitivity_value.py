import matplotlib.pyplot as plt
import numpy as np
from handlecall import handle_call


def main(args, datlst):

    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)

    dx = 1e-2

    for _ in range(args.repeat):
        for env, optimal, dpc, timevec, sample, descr, _ in datlst:

            marginal_value = []
            for t in range(len(timevec)):

                env.reset()
                u = 0
                for t_ in range(len(timevec)):
                    if t_ == t:
                        _, r, _, _ = env.step(optimal.actionvec[t_] * [1 - dx, 1 - dx])
                    else:
                        _, r, _, _ = env.step(optimal.actionvec[t_])
                    u += r
                marginal_value.append((u - optimal.utility) / dx)

            ax1.plot(timevec, marginal_value, label=descr)
            ax2.plot(timevec, np.log(np.abs(marginal_value)), label=descr)
    ax1.plot(timevec, np.zeros(len(timevec)), "--g")
    plt.subplot(ax1)
    plt.xlabel("Time (years)")
    plt.ylabel("Derivative wrt action (at optimum)")
    ax1.legend()
    ax2.legend()
    plt.show()


if __name__ == "__main__":
    handle_call(main)
