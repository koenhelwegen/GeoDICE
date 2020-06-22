" Imports "

# standard
import pickle
import timeit
import time
import numpy as np
import collections
import sys
import pdb
from mpi4py import MPI
import logging

# project
cimport coordinator
cimport lib.approx.chebyshev as cheb
cimport lib.approx.chebyshev as cheb
import lib.approx.chebyshev as cheb  # necessary because of node generator
cimport tools
import lib.models.tools as modeltools
cimport lib.approx.tools as approxtools

" Dynamic Programming Coordinator"

cdef class DPCoordinator(object):
    """The Dynamic Programming Coordinator class manages the various tasks
    involved in stochastic optimization through dynamic programming. It is build
    around an environment object, and it provides functions to perform the
    optimization, basic analysis (SCC calculations), and saving and loading,
    so that the user doesn't have to keep track of the underlying value function
    approximations."""


    def __init__(self, env, cheb_degree, n_approx_nodes):
        """Initialize coordinator: set environment and parameters for
        optimization.

        Args:
            env - environment object
            cheb_degree - degree of the Chebyshev polynomials used for
                approximation of the valuefunction
            n_approx_nodes - number of Chebyshev approximation nodes used per
                dimension (list, one item per dimension).
        """

        # check
        if type(n_approx_nodes) == int:
            n_approx_nodes = [n_approx_nodes] * env.get_state_dimension()
        assert len(n_approx_nodes) == env.get_state_dimension()
        # if type(cheb_degree) == int:
        #     cheb_degree = [cheb_degree] * env.get_state_dimension()
        # assert len(cheb_degree) == env.get_state_dimension()

        #assert env.noise

        # environment information
        domain_low, domain_high = modeltools.load_domain(env)
        self.env = env
        self.n_worlds = env.get_number_of_worlds()
        self.n_approx_nodes = n_approx_nodes
        self.domain_low = domain_low
        self.domain_high = domain_high
        self.ndim = env.get_state_dimension()
        self.tmax = env.get_max_time()

        # chebyshev approximation information
        self.chebindlst = cheb.get_index_list(self.ndim, np.max(cheb_degree))
        # self.chebindlst = cheb.clean_index_list(cheb_degree, self.chebindlst)
        self.n_approx_nodes = n_approx_nodes
        self.cheb_degree = cheb_degree
        self.cheb_length = len(self.chebindlst)
        self.cheb_d = np.sum(np.array(self.chebindlst) > 0, axis=1).tolist()
        self.cheb_coef = np.zeros([self.n_worlds, env.get_max_time()+1, self.cheb_length])

        # initialize environment
        env.reset()

    "--------------------------------------------------------------------------"
    " Optimization "

    def fit(self, int from_step=-1):
        """Perform optimization (i.e., find vlaue functions), starting at
        highest unfit step."""

        if from_step == -1:
            from_step = min(self.tmax, self.highest_unfit_step()+1)

        comm = MPI.COMM_WORLD
        if comm.Get_rank() == 0:
            print('Fitting from step {}.'.format(from_step))
        for t in reversed(range(from_step)):
            self.recursion_step(t)
            self.save('results/dpc_backup.pickle')

    cpdef void recursion_step(self, int t):
        """Perform optimization for a single step."""

        # definitions
        comm = MPI.COMM_WORLD
        cdef int rank = comm.Get_rank()
        cdef int n_processes = comm.Get_size()
        cdef np.ndarray x, st
        cdef int i, sotw
        cdef double v
        cdef np.ndarray local_cheb_coef, global_cheb_coef
        cdef np.ndarray a
        t0 = timeit.default_timer()

        if rank == 0:
            logging.debug('Starting step {}'.format(t))

        # reset current cheb coefficients:
        for i in range(self.cheb_length):
            for sotw in range(self.n_worlds):
                self.cheb_coef[sotw, t, i] = 0

        # optimize value for each node and keep track of Chebyshev sums

        # note thate we fit a seperate value function for each sotw
        for sotw in range(0, self.n_worlds):
            if (sotw == 0 or sotw == 5) and self.env.noise and self.env.get_modelcode() == 2:
                continue
            if rank == 0:
                logging.debug('SOTW {}'.format(sotw))
            for x in cheb.node_generator(self.ndim, self.n_approx_nodes, rank, n_processes):
                a = np.array([.3, .15])  # later, recylce optimal action as initial
                st = cheb.box_to_domain(x, self.domain_low[t], self.domain_high[t])
                v, a = self.optimize_action_value(t, st, sotw, a)
                # update each Chebyshev term
                for i in range(self.cheb_length):
                    self.cheb_coef[sotw, t, i] += v * cheb.ndim_polynomial(self.chebindlst[i], x)

        if rank == 0:
            logging.debug('Beginning synchronization')

        # add up results over processors
        for sotw in range(self.n_worlds):
            local_cheb_coef = self.cheb_coef[sotw, t, :]
            global_cheb_coef = np.zeros(len(local_cheb_coef))
            comm.Allreduce(local_cheb_coef, global_cheb_coef, op=MPI.SUM)
            self.cheb_coef[sotw, t, :] = global_cheb_coef

        # apply dimensionality factor
        for i in range(self.cheb_length):
            for sotw in range(self.n_worlds):
                self.cheb_coef[sotw, t, i] = (2.0**self.cheb_d[i]) / (1.0*np.prod(self.n_approx_nodes)) * self.cheb_coef[sotw, t, i]

        if rank == 0:
            logging.debug('Finished step {} after {} seconds'.format(t, timeit.default_timer() - t0))

        return

    cpdef double valuefun(self, int t, np.ndarray st, int sotw):
        """Evaluate value function."""

        cdef double temp
        if self.env.get_modelcode() == 2 and self.env.noise:
            if sotw == 0:
                temp = sum(st[5:7])
                if temp < 2:
                    return self.valuefun(t, st, 2)
                else:
                    return self.valuefun(t, st, 3)
            elif sotw == 5:
                temp = sum(st[5:7])
                if temp < 2:
                    return self.valuefun(t, st, 7)
                else:
                    return self.valuefun(t, st, 8)

        try:
            st = approxtools.project_to_domain(st, self.domain_low[t], self.domain_high[t])
            x = cheb.domain_to_box(st, self.domain_low[t], self.domain_high[t])
            return cheb.approximation(self.chebindlst, self.cheb_coef[sotw, t,:].tolist(), x)
        except IndexError:
            print('Error: sotw {}, len cheb_coef {}'.format(sotw, len(self.cheb_coef)))
            return 0

    cpdef tuple optimize_action_value(self, int t, np.ndarray st, int sotw,
                                      np.ndarray a0):
        """Optimize node value."""
        return tools.optimize_node(self.env, t, st, sotw, a0, self.valuefun)

    def highest_unfit_step(self):
        """Find highest step for which valuefunction has not yet been fitted."""
        unfit = -1
        for t in range(self.tmax):
            fit_empty = True
            for sotw in range(self.get_n_worlds()):
                if ~np.all(self.cheb_coef[sotw, t, :] == 0):
                    fit_empty = False
            if not fit_empty:
                break
            unfit = t
        return unfit

    "--------------------------------------------------------------------------"
    " Analysis (make sure valuefunctions are already fitted)"

    def sample_forward_trajectory(self, fixed_actions=[], verbose=False):
        """Sample forward trajectory."""

        # definitions
        cdef double utility
        cdef np.ndarray st, a
        cdef list trajectory, actionvec, sotwvec
        cdef int t
        cdef r
        trajectory = []
        actionvec = []
        sotwvec = []
        predictionvec = []
        utilityvec = []

        # initialize
        st, sotw = self.env.reset()
        utility = 0

        # sample path
        for t in range(self.tmax):

            # print update
            if verbose:
                print('Step: {} | SOTW: {} | Predicted value: {} | Utilility: {}'.format(t, sotw, utility + self.valuefun(t, st, sotw), utility))

            # predict value of current state (over whole trajectory, incl past)
            predictionvec.append(utility + self.valuefun(t, st, sotw))

            # store current utility and state
            utilityvec.append(utility)
            trajectory.append(st)
            sotwvec.append(sotw)

            # store image environment (for safety check)
            img = self.env.getimage()

            # optimize action
            if t < len(fixed_actions):
                a = np.array(fixed_actions[t])
            else:
                a0 = np.array([.3, .2])
                _, a = self.optimize_action_value(t, st, sotw, a0)

            # perform safety check to see if optimization didnt change env
            if not np.array(img == self.env.getimage()).all():
                print("Image changed at time {}!!".format(t))
                print(img)
                print(self.env.getimage())
                raise ValueError
            actionvec.append(a)

            # take next step and update utility
            st, r, _, sotw = self.env.step(actionvec[t])
            utility += r

        # also add last state
        trajectory.append(st)
        sotwvec.append(sotw)

        # package & return results
        res = collections.namedtuple('sampleTrajectory', ['utility',
                                                           'trajectory',
                                                           'sotwvec',
                                                           'actionvec',
                                                           'predictionvec',
                                                           'utilityvec'])
        res.utility = utility
        res.trajectory = trajectory
        res.sotwvec = sotwvec
        res.actionvec = actionvec
        res.predictionvec = predictionvec
        res.utilityvec = utilityvec
        return res

    def estimate_SCC(self, t, st, sotw, dx=1, verbose=False):
        """Estimate SCC."""

        # ratios for a single unit of emissions over the carbon reservoirs
        a_p = 0.2173
        a = [0.2240, 0.2824, 0.2763]

        # evaluate derivatives
        u0 = self.valuefun(t, st, sotw)
        self.env.setstate(t, st, sotw)
        self.env.add_marginal_capital(dx)
        u1_capital = self.valuefun(t, self.env.getstate(), sotw)
        self.env.setstate(t, st, sotw)
        self.env.add_marginal_carbon(dx)
        u1_carbon = self.valuefun(t, self.env.getstate(), sotw)
        #u1_carbon = self.valuefun(t, st + dx*np.array([0, a_p, a[0], a[1], a[2], 0, 0]), sotw)
        scc = -1000*(u1_carbon - u0)/(u1_capital - u0)

        if verbose:
            print('DPO | Time: {} | Marginal carbon: {:.5f} | Marginal capital: {:.5f} | SCC: {:.2f}'.format(
              t, (u1_carbon-u0)/dx, (u1_capital-u0)/dx, scc))

        # return ratio
        return scc

    "--------------------------------------------------------------------------"
    " Administration "

    def print_info(self):
        print('Model dimensions: {}'.format(self.ndim))
        print('Approximation nodes: {}'.format(self.n_approx_nodes))
        print('Chebyshev degree: {}'.format(self.cheb_degree))
        print('Stepsize: {}'.format(self.env.get_stepsize()))
        print('Experiment: {}'.format(self.env.get_experiment()))
        print('Noise: {}'.format(self.env.noise))
        print('')

    def get_cheb_degree(self):
        return self.cheb_degree

    def get_n_approx_nodes(self):
        return self.n_approx_nodes

    def get_n_worlds(self):
        return len(self.cheb_coef)

    def filename(self):
        if len(np.unique(self.n_approx_nodes)) == 1:
            nodestring = str(np.unique(self.n_approx_nodes)[0])
        else:
            nodestring = ''.join([str(n) for n in self.n_approx_nodes])

        filename = 'noise{}step{}experiment{}degree{}nodes{}'.format(
          self.env.noise,
          self.env.get_stepsize(),
          self.env.get_experiment(),
          self.cheb_degree,
          nodestring)

        return filename

    def save(self, filename=None, verbose=False):
        "Save object to file using pickle."

        # basic info
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        if filename is None:
            filename = 'data/{}/dpc/{}'.format(
              self.env.get_modeltype(),
              self.filename()
            )

        if rank == 0:
            with open(filename, 'wb') as f:
                pickle.dump(self, f)
            if verbose:
                print('Saved DPC in {}'.format(filename))

    def get_domain(self):
        return self.domain_low, self.domain_high

    def update_env(self, env):
        self.env = env


def load(env=None, degree=None, nodes=None, filename=None, allow_not_found=False):

    updatemodel = False

    if filename is None:
        updatemodel = True
        if len(np.unique(nodes)) == 1:
            nodestring = str(np.unique(nodes)[0])
        else:
            nodestring = ''.join([str(n) for n in nodes])

        filename = 'data/{}/dpc/noise{}step{}experiment{}degree{}nodes{}'.format(
          env.get_modeltype(), env.noise, env.get_stepsize(), env.get_experiment(),
          degree, nodestring)

    comm = MPI.COMM_WORLD
    if comm.Get_rank() == 0:
        print('Loading coordinator from {}...'.format(filename))

    try:
        with open(filename, 'rb') as f:
            x = pickle.load(f)
    except FileNotFoundError:
        if comm.Get_rank() == 0:
            if allow_not_found:
                print('Did not find {}.'.format(filename))
                x = None
            else:
                print('Problem loading data for noise={}, stepsize={}, experiment={}, degree={}, nodes={}.'.format(
                  env.noise, env.get_stepsize(), env.get_experiment(), degree, nodes))
                print('Please make sure this data exists.')
                sys.exit()

    # update modeltype
    if x is not None and updatemodel:
        x.update_env(env)

        dl, dh = modeltools.load_domain(x.env)
        dpc_dl, dpc_dh = x.get_domain()
        if not (np.array_equal(dl, dpc_dl) and np.array_equal(dh, dpc_dh)):
            print('WARNING - domains used in DPC do not match current boundaries for this model.')

        if env.get_number_of_worlds() != x.get_n_worlds():
            print('WARNING - mismatch in number of worlds: loaded dpc optimized for {} worlds, but environment has {}.'.format(
              x.get_n_worlds(),
              env.get_number_of_worlds()
            ))

    return x
