# standard imports
import matplotlib.pyplot as plt
import numpy as np

# project imports
from handlecall import handle_call


def main(args, datlst):

    for env, optimal, dpc, timevec, _, _, _ in datlst:

        dx = 1e-3

        plt.plot(timevec, np.zeros(len(timevec)), "--g")

        for var in range(env.get_state_dimension()):

            derivative_error = []

            env.reset()
            u = 0
            for t in range(len(timevec)):

                # actual state
                st = env.getstate()

                # marginal state
                env.add_marginal_variable(var, dx=dx)
                st_ = env.getstate()

                # true derivative
                u_ = 0
                # continue optimal trajectory (from marginal state)
                for t_ in range(t, len(timevec)):
                    _, r, _, _ = env.step(optimal.actionvec[t_])
                    u_ += r
                true_derivative = (u_ + u - optimal.utility) / dx

                # DP derivative
                dpc_derivative = (dpc.valuefun(t, st_, 0) - dpc.valuefun(t, st, 0)) / dx
                derivative_error.append(100 * dpc_derivative / true_derivative - 100)

                # take step
                env.setstate(t, st, 0)
                _, r, _, _ = env.step(optimal.actionvec[t])
                u += r

            plt.plot(timevec, derivative_error, label="var{}".format(var))

    plt.xlabel("Time (years)")
    plt.ylabel("Error in derivative (% difference)")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    handle_call(main)
