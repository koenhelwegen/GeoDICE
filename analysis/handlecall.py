# standard imports
import argparse
from collections import namedtuple

# project imports
import lib.models.tools as modeltools
import lib.dp.coordinator as coordinator
import sampleanalysis.tools as sampletools


def handle_call(callback, dpc_required=True, load_anything=True):
    # create parser
    parser = argparse.ArgumentParser()

    # ARGUMENTS

    # model, experiment and optimization settings
    parser.add_argument("-s", "--stepsize", nargs="+", type=int, default=[10])
    parser.add_argument("-e", "--experiment", nargs="+", type=int, default=[21])
    parser.add_argument("-d", "--degree", nargs="+", type=int, default=[4])
    parser.add_argument("-n", "--nodes", nargs="+", type=int, default=[5])

    # noise
    parser.add_argument("--noise", dest="noise", action="store_true")
    parser.add_argument("--no-noise", dest="noise", action="store_false")
    parser.set_defaults(noise=True)

    # output flags and settings (to finetune plotting and printing behaviour)
    parser.add_argument("-c", "--center", type=int, default=1)
    parser.add_argument("-p", "--print_result", type=int, default=0)
    parser.add_argument("-v", "--verbose", type=int, default=0)
    parser.add_argument("-m", "--modeltype", default="lrgd2")
    parser.add_argument("--dimension", nargs="+", type=int, default=[-1])
    parser.add_argument("--sotw", nargs="+", type=int, default=[0])
    parser.add_argument("-t", "--time", type=int, default=10)
    parser.add_argument("-f", "--fixed", type=int, default=0)
    parser.add_argument("-r", "--repeat", type=int, default=1)
    parser.add_argument("--savefile", default=None)
    parser.add_argument("--savedata", default=None)

    # plot full interval
    parser.add_argument("--full", dest="fulltime", action="store_true")
    parser.set_defaults(fulltime=False)

    # also plot domain
    parser.add_argument("--domain", dest="plot_domain", action="store_true")
    parser.set_defaults(plot_domain=False)

    # use data for deterministic policy
    parser.add_argument("--detpol", dest="detpol", action="store_true")
    parser.set_defaults(detpol=False)

    # use data for deterministic policy
    parser.add_argument("--nopol", dest="nopol", action="store_true")
    parser.set_defaults(nopol=False)

    # use data for deterministic policy
    parser.add_argument(
        "--suppress_legend", dest="suppress_legend", action="store_true"
    )
    parser.set_defaults(suppress_legend=False)

    # parse
    args = parser.parse_args()

    Data = namedtuple(
        "data", ["env", "optimal", "dpc", "timevec", "sample", "descr", "longdescr"]
    )

    datlst = []
    if load_anything:
        print("Loading data...")
        for stepsize in args.stepsize:
            for experiment in args.experiment:
                for degree in args.degree:
                    for nodes in args.nodes:

                        # create environment
                        env = modeltools.load_model(
                            modeltype=args.modeltype,
                            stepsize=stepsize,
                            noise=args.noise,
                            experiment=experiment,
                        )
                        # load data
                        optimal = modeltools.load_optimal_pathway(env)
                        dpc = coordinator.load(env, degree, nodes, allow_not_found=True)
                        timevec = env.get_timevec()

                        if dpc is None and dpc_required:
                            continue

                        # sample trajectory
                        samplevec = []
                        if dpc is not None:
                            if dpc.highest_unfit_step() == -1:
                                if args.repeat == 1:
                                    sample = dpc.sample_forward_trajectory(
                                        fixed_actions=optimal.actionvec[: args.fixed]
                                    )
                                    samplevec = [sample]
                                else:
                                    samplevec = sampletools.load_sample(
                                        dpc, args.repeat
                                    )

                        if samplevec == []:
                            samplevec = None

                        descr = ""
                        if len(args.stepsize) > 1:
                            descr = descr.join("s{} ".format(stepsize))
                        if len(args.experiment) > 1:
                            descr = descr.join("e{} ".format(experiment))
                        if len(args.degree) > 1:
                            descr += "d{} ".format(degree)
                        if len(args.nodes) > 1:
                            descr += "n{}".format(nodes)

                        longdescr = "Stepsize: {} | Experiment: {} | Degree: {} | Nodes: {}".format(
                            stepsize, experiment, degree, nodes
                        )

                        dat = Data(
                            env, optimal, dpc, timevec, samplevec, descr, longdescr
                        )

                        datlst.append(dat)
    else:
        datlst = []

    # callback
    callback(args, datlst)
