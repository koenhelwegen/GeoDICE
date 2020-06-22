from handlecall import handle_call
import matplotlib.pyplot as plt


def main(args, datlst):

    ax_co2 = plt.subplot(121)
    plt.xlabel("Time (year)")
    plt.ylabel("Atmospheric CO2 (ppm)")

    ax_temp = plt.subplot(122)
    plt.xlabel("Time (year)")
    plt.ylabel("GMST (degrees Celcius above pre-industrial)")

    samplevec = []

    for _ in range(args.repeat):
        for env, optimal, dpc, timevec, _, descr, _ in datlst:

            assert env.get_modeltype() == "lrgeodice"

            sample = dpc.sample_forward_trajectory()

            # plot trajectories
            ax_co2.plot(
                timevec,
                [sum(st[1:5]) for st in sample.trajectory[:-1]],
                "-b",
                alpha=0.1,
            )
            ax_temp.plot(
                timevec,
                [sum(st[5:7]) for st in sample.trajectory[:-1]],
                "-b",
                alpha=0.1,
            )

            samplevec.append(sample)

    for sample in samplevec:
        # find tipping point
        if 2 in sample.sotwvec or 3 in sample.sotwvec:
            if 2 in sample.sotwvec:
                tip_t = sample.sotwvec.index(2)
            else:
                tip_t = sample.sotwvec.index(3)

            ax_co2.plot(
                timevec[tip_t], sum(sample.trajectory[tip_t][1:5]), ".r", alpha=0.2
            )
            ax_temp.plot(
                timevec[tip_t], sum(sample.trajectory[tip_t][5:7]), ".r", alpha=0.2
            )

    plt.show()


if __name__ == "__main__":
    handle_call(main)
