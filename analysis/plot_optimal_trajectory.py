# standard imports
import matplotlib.pyplot as plt
from handlecall import handle_call


def main(args, datlst):

    for env, optimal, dpc, timevec, sample, descr, _ in datlst:

        for ii in range(env.get_state_dimension()):
            plt.subplot(421 + ii)
            plt.plot(timevec, [st[ii] for st in optimal.trajectory[:-1]])

        assert not env.noise

        sotwvec = []
        _, sotw = env.reset()
        for t in range(len(timevec)):
            sotwvec.append(sotw)
            _, _, _, sotw = env.step(optimal.actionvec[t])

        plt.subplot(421 + ii + 1)
        plt.plot(timevec, sotwvec)

    plt.show()


if __name__ == "__main__":
    handle_call(main, False)
