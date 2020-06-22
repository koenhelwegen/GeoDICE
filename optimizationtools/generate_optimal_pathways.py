import lib.models.tools as modeltools
from scipy import interpolate
import numpy as np
import math
from analysis.handlecall import handle_call


def generate_optimal_pathway(args, _):

    modeltype = args.modeltype
    stepsizevector = args.stepsize
    experimentvec = args.experiment

    # make sure stepsize is from high to low
    stepsizevector.sort()
    stepsizevector.reverse()
    for experiment in experimentvec:
        for ii in range(0, len(stepsizevector)):

            # define model
            stepsize = stepsizevector[ii]
            env = modeltools.load_model(
                modeltype, stepsize=stepsize, noise=False, experiment=experiment
            )

            # get initial guess by interpolating previous optimal policy
            if ii > 0:

                # load previous data
                env_ = modeltools.load_model(
                    modeltype,
                    stepsize=stepsizevector[ii - 1],
                    noise=False,
                    experiment=experiment,
                )
                timevec_ = env_.get_timevec()
                timevec = env.get_timevec()
                opt_ = modeltools.load_optimal_pathway(env_)
                a0_ = [a[0] for a in opt_.actionvec]
                a1_ = [a[1] for a in opt_.actionvec]

                # get interpolations
                f0 = interpolate.interp1d(timevec_, a0_)
                f1 = interpolate.interp1d(timevec_, a1_)

                # interpolate (skip last steps to ensure there's no extrapolation)
                skip_steps = math.ceil(stepsizevector[ii - 1] / stepsize)
                a0 = f0(timevec[:-skip_steps]).tolist()
                a1 = f1(timevec[:-skip_steps]).tolist()
                for _ in range(skip_steps):
                    a0.append(a0[-1])
                    a1.append(a1[-1])

                # zip initialguess to format suitable for optimization
                initialguess = np.array(list(zip(a0, a1))).reshape(-1, 1)

            else:
                initialguess = None

            if stepsize > 1 or initialguess is None:
                modeltools.generate_optimal_pathway(env, initialguess)
            else:
                # use initial guess for step 1, as optimization otherwise fails
                modeltools.generate_optimal_pathway(env, initialguess, True)


if __name__ == "__main__":
    handle_call(generate_optimal_pathway, False, False)
