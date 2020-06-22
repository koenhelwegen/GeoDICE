""" Run optimization. Call this script using something like:

        mpiexec -n 4 python run_optimization

    to run in parellel.
"""
from mpi4py import MPI

# standard
import timeit
import datetime
import logging

# project
import lib.dp.coordinator as coordinator
import lib.models.tools as modeltools


"""# PREPERATION #"""

# logfile
logging.basicConfig(level=logging.DEBUG, filename="temp.log", filemode="w")

# experiment settings
stepsize = 10
experiment = 3000
modeltype = "lrgd2"
noise = True

# dp settings
cheb_degree = 4
n_cheb_nodes = 5

# create environment
env = modeltools.load_model(
    modeltype=modeltype, stepsize=stepsize, experiment=experiment, noise=noise
)

# check boundaries
domain_low, domain_high = modeltools.load_domain(env)
assert len(domain_high) == env.get_max_time() + 1

# get MPI info
comm = MPI.COMM_WORLD

# print update
if comm.Get_rank() == 0:
    print(
        "Starting optimization with model {}, experiment {}, stepsize {}, Chebyshev degree {} and approximation nodes per dimension {}.".format(
            modeltype, experiment, stepsize, cheb_degree, n_cheb_nodes
        )
    )
    logging.info(
        "Starting optimization with model {}, experiment {}, stepsize {}, Chebyshev degree {} and approximation nodes per dimension {}.".format(
            modeltype, experiment, stepsize, cheb_degree, n_cheb_nodes
        )
    )

"""# OPTIMIZATION #"""
t0 = timeit.default_timer()

dpc = coordinator.DPCoordinator(env, cheb_degree, n_cheb_nodes)
dpc.fit()
dpc.save(verbose=True)

t1 = timeit.default_timer()

# print update
if comm.Get_rank() == 0:
    print(
        "Program finished after {:.2f}sec on {} processors.".format(
            t1 - t0, comm.Get_size()
        )
    )
    print("Datetime: ", str(datetime.datetime.now()))
    print("Total CPU hours: {:.2f}".format((t1 - t0) * comm.Get_size() / 3600))
