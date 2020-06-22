import lib.models.tools as modeltools
from analysis.handlecall import handle_call


def generate_domain(args, _):

    for experiment in args.experiment:
        for stepsize in args.stepsize:
            env = modeltools.load_model(
                modeltype=args.modeltype,
                stepsize=stepsize,
                experiment=experiment,
                noise=False,
            )
            # if optimal_experiment is not -1:
            #     env_opt = modeltools.load_model(modeltype=modeltype, stepsize=stepsize, experiment=optimal_experiment, noise=False)
            # else:
            #     env_opt = None
            # modeltools.generate_domain(env, env_opt)
            modeltools.generate_domain(env)


if __name__ == "__main__":
    handle_call(generate_domain, False)
