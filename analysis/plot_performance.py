# standard imports
import matplotlib.pyplot as plt
import numpy as np
from handlecall import handle_call


def main(args, datlst):

    for _ in range(args.repeat):

        for env, optimal, dpc, timevec, sample, descr, _ in datlst:

            performance = []
            for t in range(len(timevec) - 1):
                # bau performance
                bau = 0
                env.reset()
                for t_ in timevec:
                    if t_ < t:
                        _, r, _, _ = env.step(optimal.actionvec[t_])
                    else:
                        _, r, _, _ = env.step(np.array([0, 0]))
                    bau += r

                res = dpc.sample_forward_trajectory(optimal.actionvec[:t])
                performance.append((res.utility - bau) / (optimal.utility - bau))
                if t < args.print_result or args.print_result == -1:
                    print(
                        "{} | {}".format(
                            timevec[t], (res.utility - bau) / (optimal.utility - bau)
                        )
                    )
            plt.plot(timevec[:-1], performance, "-", label="{}".format(descr))

    plt.xlabel("Time (years)")
    plt.ylabel("Relative performance")
    if descr != "":
        plt.legend()
    plt.show()


if __name__ == "__main__":
    handle_call(main)
