# standard imports
import matplotlib.pyplot as plt
import numpy as np

# project imports
from handlecall import handle_call


def plot_error(args, datlst):

    # prepare input
    print_result = args.print_result

    for env, optimal, dpc, timevec, sample, descr, _ in datlst:

        # plot error
        plt.plot(
            timevec,
            np.log(abs(np.array(sample.predictionvec) - sample.utility)),
            ".",
            label=descr,
        )

        # print error
        if print_result != 0:
            for ii in range(len(sample.predictionvec)):
                print("Time {} | Prediction {}".format(ii, sample.predictionvec[ii]))
                if ii + 1 == print_result:
                    break
            print("Actual utility: {}".format(sample.utility))

    # decorate
    plt.xlabel("Time (years)")
    plt.ylabel("Log error in prediction")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    handle_call(plot_error)
