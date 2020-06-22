# standard imports
import matplotlib.pyplot as plt
from handlecall import handle_call


def plot_samplepath(args, datlst):

    for env, optimal, dpc, timevec, samplevec, descr, _ in datlst:

        for sample, ii in zip(samplevec, list(range(len(samplevec)))):

            # plot actions
            if ii == 0:
                lines = plt.plot(
                    timevec,
                    [a[0] for a in sample.actionvec],
                    "-",
                    label="Mitigation ({})".format(descr),
                    alpha=0.3,
                )
                plt.plot(
                    timevec,
                    [a[1] for a in sample.actionvec],
                    "--",
                    label="SRM ({})".format(descr),
                    color=lines[0].get_c(),
                    alpha=0.3,
                )
            else:
                plt.plot(
                    timevec,
                    [a[0] for a in sample.actionvec],
                    "-",
                    color=lines[0].get_c(),
                    alpha=0.3,
                )
                plt.plot(
                    timevec,
                    [a[1] for a in sample.actionvec],
                    "--",
                    color=lines[0].get_c(),
                    alpha=0.3,
                )

            if args.print_result > 0:
                for a, step in zip(
                    sample.actionvec[: args.print_result], range(args.print_result)
                ):
                    print("{} | ({}, {})".format(timevec[step], a[0], a[1]))

    lines = plt.plot(
        timevec, [a[0] for a in optimal.actionvec], ".r", label="Mitigation (optimal)"
    )
    plt.plot(
        timevec,
        [a[1] for a in optimal.actionvec],
        "x",
        label="SRM (optimal)",
        color=lines[0].get_c(),
    )

    if not args.fulltime:
        plt.xticks([2015, 2050, 2100, 2150, 2200, 2250, 2300])
        plt.xlim([2015, 2305])

    plt.legend()
    plt.xlabel("Time (years)")
    plt.ylabel("Mitigation (ratio of output)\nSRM (100 T S/yr)")

    plt.show()


if __name__ == "__main__":
    handle_call(plot_samplepath)
