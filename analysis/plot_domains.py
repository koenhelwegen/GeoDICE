import lib.models.tools as modeltools
import matplotlib.pyplot as plt
import numpy as np
from handlecall import handle_call
import sampleanalysis.tools as sampletools


def plot_domains(args, datlst):

    env, optimal, dpc, timevec, _, _, _ = datlst[0]

    # generate axes
    ax = []
    counter = 0
    for ii in range(env.get_state_dimension()):
        for jj in range(2):
            ax.append(plt.subplot(env.get_state_dimension(), 2, 1 + counter))
            plt.subplot(ax[-1])
            if jj == 0:
                plt.ylabel("Dim {}".format(ii))
            else:
                plt.ylabel("Dim {}\nLogarithmic".format(ii))
            if ii == env.get_state_dimension() - 1:
                plt.xlabel("Time (years)")
            counter += 1

    for env, optimal, dpc, timevec, _, _, _ in datlst:

        domain_low, domain_high = modeltools.load_domain(env)

        # loop over dimensions
        counter = 0
        for ii in range(env.get_state_dimension()):

            # REGULAR
            # lower bound
            ax[counter].plot(timevec, [low[ii] for low in domain_low[:-1]], "--r")
            # upper bounds
            ax[counter].plot(timevec, [high[ii] for high in domain_high[:-1]], "--r")
            # sample path
            if args.noise and args.repeat > 0:
                if dpc is not None:
                    samplevec = sampletools.load_sample(dpc, args.repeat)
                    for sample in samplevec:
                        ax[counter].plot(
                            timevec,
                            [st[ii] for st in sample.trajectory[:-1]],
                            "-b",
                            alpha=0.05,
                        )

            # optimal pathway
            ax[counter].plot(timevec, [st[ii] for st in optimal.trajectory[:-1]], "-r")
            counter += 1

            # LOGARITHMIC
            # optimal pathway
            ax[counter].plot(
                timevec,
                [np.log(max(0.00001, st[ii])) for st in optimal.trajectory[:-1]],
                "-r",
            )
            # lower bound
            ax[counter].plot(
                timevec,
                [np.log(max(0.00001, low[ii])) for low in domain_low[:-1]],
                "--r",
            )
            # upper bounds
            ax[counter].plot(
                timevec,
                [np.log(max(0.00001, high[ii])) for high in domain_high[:-1]],
                "--r",
            )
            counter += 1
            # sample path
            # if args.noise and args.repeat > 0:
            #     samplevec = sampletools.load_sample(dpc, args.repeat)
            #     for sample in samplevec:
            #         ax[counter].plot(timevec, [np.log(max(.00001, st[ii])) for st in sample.trajectory[:-1]], '-b', alpha=.3)

    plt.show()


if __name__ == "__main__":
    handle_call(plot_domains, False)
