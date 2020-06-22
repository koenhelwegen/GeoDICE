# standard imports
import matplotlib.pyplot as plt

# project imports
import lib.models.tools as modeltools
from handlecall import handle_call


def main(args, datlst):

    for env, optimal, dpc, timevec, samplevec, descr, _ in datlst:
        for sample in samplevec:
            bnds_low, bnds_high = modeltools.load_domain(env)
            for ii in range(env.get_state_dimension()):
                plt.subplot(421 + ii)
                plt.plot(
                    timevec, [st[ii] for st in sample.trajectory[:-1]], "-b", alpha=0.2
                )
                plt.ylabel("Dimension {}".format(ii))
                plt.xlabel("Time (years)")

    if args.plot_domain:
        for ii in range(env.get_state_dimension()):
            plt.subplot(421 + ii)
            plt.plot(timevec, [st[ii] for st in optimal.trajectory[:-1]], "-r")
            plt.plot(timevec, [l[ii] for l in bnds_low[:-1]], "--r")
            plt.plot(timevec, [h[ii] for h in bnds_high[:-1]], "--r")

    plt.show()


if __name__ == "__main__":
    handle_call(main)
