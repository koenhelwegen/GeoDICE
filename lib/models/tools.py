from scipy.optimize import minimize
import timeit
import numpy as np
import pickle
import collections
import lib.models.lrgd2 as lrgd2
from collections import namedtuple


def generate_optimal_pathway(env, initialguess=None, skip_optimization=False):
    """Optimize pathway for deterministic model and store result."""

    # basic info
    stepsize = env.get_stepsize()
    filename = "data/{}/optimal/step{}experiment{}.pickle".format(
        env.get_modeltype(), stepsize, env.get_experiment()
    )
    tmax = env.get_max_time()
    action_dim = len(env.get_action_bounds())

    # update user
    print(
        "Optimizing experiment {}, model {} with policy-steps of {} years...".format(
            env.get_experiment(), env.get_modeltype(), env.get_stepsize()
        )
    )

    # check environment is deterministic
    assert env.noise == 0

    # find bounds over course of trajectory
    bnds = []
    env.reset()
    while True:
        bnds.extend(env.get_action_bounds())
        _, _, done, _ = env.step(np.zeros(2))
        if done:
            break
    # bnds = env.get_action_bounds() * tmax

    # define map from all actions to -utility (for minimization)
    def fullrun(u):
        env.reset()
        utility = 0
        for i in range(tmax):
            _, reward, _, _ = env.step(u[i * action_dim : (i + 1) * action_dim])
            utility += reward
        return -utility

    # if necessary, set/load initial guess
    if initialguess is None:
        initialguess = np.zeros(len(bnds)) + 0.3
    elif initialguess is -1:
        with open(filename, "rb") as f:
            res = pickle.load(f)
        initialguess = res[1]

    # perform minimization
    if not skip_optimization:
        res = minimize(
            fullrun,
            initialguess,
            bounds=bnds,
            method="slsqp",
            options={"ftol": 1e-10, "maxiter": 500},  # , 'disp':True}
        )
    else:
        print("Skipping optimzation...")
        res = namedtuple("result", ["x", "success", "fun"])
        res.x = initialguess
        res.success = True
        res.fun = fullrun(initialguess)

    # check if optimization was succesful
    if not res.success:
        print("WARNING: Optimization not successful.")
        return
    else:
        print("Policy succesfully optimized!")

    # convert action to convenient format
    x = np.array(list(zip(res.x[0::2], res.x[1::2])))

    # generate optimal pathway
    traj = []
    env.reset()
    for t in range(env.get_max_time()):
        traj.append(env.getstate())
        env.step(x[t])
    traj.append(env.getstate())

    res = tuple([-res.fun, x, traj])

    with open(filename, "wb") as f:
        pickle.dump(res, f)
        print("Optimal pathway stored in {}".format(filename))


def load_optimal_pathway(env):
    """Load optimal pathway."""

    # get filename
    filename = "data/{}/optimal/step{}experiment{}.pickle".format(
        env.get_modeltype(), env.get_stepsize(), env.get_experiment()
    )

    # load and return
    # print('Loading optimal policy from {}'.format(filename))
    with open(filename, "rb") as f:
        res = pickle.load(f)

    Optimal = collections.namedtuple("optimizedPathway", "utility actionvec trajectory")
    optimal = Optimal(res[0], res[1], res[2])

    return optimal


def generate_domain(env, env_opt=None):

    # read out model info
    stepsize = env.get_stepsize()
    experiment = env.get_experiment()
    modeltype = env.get_modeltype()
    statedimension = env.get_state_dimension()
    tmax = env.get_max_time()
    filename = "data/{}/domain/step{}experiment{}.pickle".format(
        env.get_modeltype(), stepsize, experiment
    )

    if env_opt is not None:
        env = env_opt

    # get optimal pathway
    optimal = load_optimal_pathway(env)
    traj_opt = optimal.trajectory

    # get high-abatement pathway
    traj_ha = []
    env.reset()
    for t in range(tmax):
        traj_ha.append(env.getstate())
        env.step(np.array([max(env.get_action_bounds()[0]), 0]))
    traj_ha.append(env.getstate())

    # get low-abatement pathway
    traj_la = []
    env.reset()
    for t in range(tmax):
        traj_la.append(env.getstate())
        if t * stepsize < 30:
            a = [0, 0]
        elif t > env.get_max_time() - (40 / stepsize):
            a = [0.8, 0]
            if experiment in [22]:
                a = [0.0, 0.0]
        else:
            a = [
                min(
                    min(0.004 * (t * stepsize - 30), 0.99),
                    max(env.get_action_bounds()[0]),
                ),
                0,
            ]
        env.step(np.array(a))
    traj_la.append(env.getstate())

    domain_low = np.zeros([env.get_max_time() + 1, statedimension]).tolist()
    domain_high = np.zeros([env.get_max_time() + 1, statedimension]).tolist()
    for t in range(env.get_max_time() + 1):
        if modeltype == "lrgd2":
            # K
            domain_low[t][0] = 0.7 * traj_opt[t][0]
            domain_high[t][0] = 1.3 * traj_opt[t][0]
            if experiment in [22]:
                if t == 0:
                    print("Generating domain for lazy-abatement scenario...")

                # C
                domain_low[t][1] = 300  # .5 * traj_ha[t][1]
                domain_high[t][1] = max(1.5 * traj_la[t][1], 600)
                domain_low[t][2] = 0  # .2 * traj_ha[t][2]
                domain_high[t][2] = max(1.5 * traj_la[t][2], 250)
                domain_low[t][3] = 0  # .2 * traj_ha[t][3]
                domain_high[t][3] = max(1.5 * traj_la[t][3], 250)
                domain_low[t][4] = 0  # .2 * traj_ha[t][4]
                domain_high[t][4] = max(1.5 * traj_la[t][4], 50)
                # if t > env.get_max_time() - 3:
                #     domain_high[t][4] += 50 * (t-env.get_max_time()+30/stepsize)
                # T
                domain_low[t][5:7] = [-0.5, -1.2]
                domain_high[t][5] = min(max(2.5, 2 * traj_la[t][5]), 10)
                domain_high[t][6] = min(max(2, 2 * traj_la[t][6]), 12)

            elif experiment == 21:

                if t == 0:
                    print("Generating domain for unconstrained scenario...")
                # C
                domain_low[t][1] = 300  # .5 * traj_ha[t][1]
                domain_high[t][1] = max(1.2 * traj_la[t][1], 0)
                domain_low[t][2] = 0  # .2 * traj_ha[t][2]
                domain_high[t][2] = max(1.6 * traj_la[t][2] + 20, 200)
                domain_low[t][3] = -20  # .2 * traj_ha[t][3]
                domain_high[t][3] = max(1.6 * traj_la[t][3] + 10, 50)
                domain_low[t][4] = -3  # .2 * traj_ha[t][4]
                domain_high[t][4] = min(max(1.4 * traj_la[t][4], 10), 20)
                # if t > env.get_max_time() - 3:
                #     domain_high[t][4] += 50 * (t-env.get_max_time()+30/stepsize)
                # T
                domain_low[t][5:7] = [-0.5, -1.2]
                domain_high[t][5] = min(max(2.5, 2 * traj_la[t][5]), 5)
                domain_high[t][6] = min(max(2, 2 * traj_la[t][6]), 7)

            else:
                if t == 0:
                    print("Generating domain for abatement-only scenario...")
                # C
                domain_low[t][1] = 0.5 * traj_ha[t][1]
                domain_high[t][1] = max(min(1.2 * traj_la[t][1], 2000), 200)
                domain_low[t][2] = 0
                domain_high[t][2] = max(min(1.6 * traj_la[t][2], 1000), 100)
                domain_low[t][3] = 0
                domain_high[t][3] = max(min(1.6 * traj_la[t][3], 150), 50)
                domain_low[t][4] = 0
                domain_high[t][4] = min(max(1.4 * traj_la[t][4], 10), 20)
                # if t > env.get_max_time() - 3:
                #     domain_high[t][4] += 50 * (t-env.get_max_time()+3)
                # T
                domain_low[t][5:7] = [-0.5, -1.2]
                domain_high[t][5] = min(max(3, 2 * traj_la[t][5]), 5)
                domain_high[t][6] = min(max(3, 2 * traj_la[t][6]), 5)
            # domain_low[t][5:7] = [0, -1.5]
            # domain_high[t][5] = 0#max(2.5, 2*traj_la[t][5])
            # domain_high[t][6] = max(2, 2*traj_la[t][6])
        else:
            raise ValueError("Domain generation: Unknown model")

    domain_low = [np.array(x) for x in domain_low]
    domain_high = [np.array(x) for x in domain_high]

    # correct upper bounds for experiment 22
    if experiment == 22:
        env2 = load_model(
            modeltype=modeltype, noise=False, experiment=20, stepsize=stepsize
        )

        _, domain_high2 = load_domain(env2)
        for t in range(len(domain_high)):
            for ii in range(len(domain_high[t])):
                domain_high[t][ii] = max(domain_high[t][ii], domain_high2[t][ii])

    with open(filename, "wb") as f:
        pickle.dump((domain_low, domain_high), f)
        print("Saved domain to {}".format(filename))

    return


def load_domain(env):
    stepsize = env.get_stepsize()
    experiment = env.get_experiment()
    filename = "data/{}/domain/step{}experiment{}.pickle".format(
        env.get_modeltype(), stepsize, experiment
    )

    with open(filename, "rb") as f:
        domain_low, domain_high = pickle.load(f)

    return domain_low, domain_high


def scc(env, t_scc, verbose=False, dx=1):

    # check input
    assert not env.noise

    # load optimal pathway
    optimal = load_optimal_pathway(env)

    # get baseline
    # u0 = optimal.utility
    u0 = 0
    sotwvec0 = []
    _, sotw = env.reset()
    for t in range(len(optimal.actionvec)):
        sotwvec0.append(sotw)
        _, r, _, sotw = env.step(optimal.actionvec[t])
        u0 += r

    # get marginal effect of carbon
    u1_carbon = 0
    sotwvec = []
    _, sotw = env.reset()
    for t in range(len(optimal.actionvec)):
        sotwvec.append(sotw)
        if t == t_scc:
            env.add_marginal_carbon(-dx)
        _, r, _, sotw = env.step(optimal.actionvec[t])
        u1_carbon += r

    if sotwvec != sotwvec0:
        print("WARNING: sotw vectors differ (carbon)")

    # get marginal effect of capital
    u1_capital = 0
    sotwvec = []
    _, sotw = env.reset()
    for t in range(len(optimal.actionvec)):
        sotwvec.append(sotw)
        if t == t_scc:
            env.add_marginal_capital(-dx)
        _, r, _, sotw = env.step(optimal.actionvec[t])
        u1_capital += r

    if sotwvec != sotwvec0:
        print("WARNING: sotw vectors differ (capital)")
        print(
            "\t First sotw switch after {} steps (default: {})".format(
                sotwvec.count(0), sotwvec0.count(0)
            )
        )

    scc_ = -1000 * (u1_carbon - u0) / (u1_capital - u0)

    if verbose:
        # print('NLO | Time: {} | u0: {} | u1_carbon: {} | u1_capital: {} | dx: {}'.format(t_scc, u0, u1_carbon, u1_capital, dx))
        print(
            "NLO | Time: {} | Marginal carbon: {:.5f} | Marginal capital: {:.5f} | SCC: {:.2f}".format(
                t_scc, (u1_carbon - u0) / dx, (u1_capital - u0) / dx, scc_
            )
        )

    return scc_


def load_model(modeltype, stepsize, noise, experiment):
    if modeltype == "lrgd2":
        return lrgd2.LrGeoDice(stepsize=stepsize, noise=noise, experiment=experiment)

    else:
        raise ValueError("Unknown modeltype")
