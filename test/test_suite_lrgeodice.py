"""Test suite for dp algorithm."""

import numpy as np
import math
import lib.models.lrgd2 as lrgeodice
import lib.models.tools as modeltools
import lib.approx.chebyshev as cheb
import lib.dp.qfunction as qfunction
import lib.dp.tools as dptools
import lib.dp.coordinator as coordinator
import timeit
import pickle

env = lrgeodice.LrGeoDice(stepsize=25, experiment=20)
senv = lrgeodice.LrGeoDice(stepsize=25, experiment=20, noise=True)
modeltools.generate_optimal_pathway(env)
modeltools.generate_domain(env)


"### MODEL ###"


def test_import_model():
    """Test the cython version of the model."""

    env.reset()

    assert len(env.step(np.array([0.2, 0.2]))) == 4

    return 0


"### CHEBYSHEV ###"


def test_chebyshev1DPolynomial():
    """Test one-dimensional Chebyshev polynomial."""
    x = np.random.rand() * 2 - 1
    assert cheb.simple_polynomial(0, x) == 1
    np.testing.assert_almost_equal(cheb.simple_polynomial(1, x), x)
    np.testing.assert_almost_equal(cheb.simple_polynomial(2, x), 2 * x * x - 1)


def test_chebyshevNDimPolynomial():
    """Test n-dimensional Chebyshev polynomial."""
    a = cheb.ndim_polynomial([3, 4], np.array([0.7, 0.11]))
    c = np.zeros([5, 5])
    c[3, 4] = 1
    d = np.polynomial.chebyshev.chebval2d(0.7, 0.11, c)
    b = cheb.simple_polynomial(3, 0.7) * cheb.simple_polynomial(4, 0.11)
    np.testing.assert_almost_equal(a, b)
    np.testing.assert_almost_equal(a, d)


def test_chebyshevIndexList():
    """Test generation of Chebyshev indices."""
    lst = cheb.get_index_list(3, 5)
    assert len(lst) == math.factorial(5 + 3) / math.factorial(5) / math.factorial(3)
    cheb_d = np.sum(np.array(lst) > 0, axis=1).tolist()
    for alpha, d in zip(lst, cheb_d):
        assert len(alpha) == 3
        assert sum(alpha) <= 5

        assert sum(np.array(alpha) > 0) == d
        assert lst.count(alpha) == 1


def test_chebyshevApproximation():
    """Test Chebyshev approximation function."""
    lst = cheb.get_index_list(2, 5)
    coef = np.random.rand(len(lst)).tolist()
    c = np.zeros([6, 6])
    for alpha, c_alpha in zip(lst, coef):
        c[alpha[0], alpha[1]] = c_alpha
    x = np.random.rand(2)
    a = np.polynomial.chebyshev.chebval2d(x[0], x[1], c)
    b = cheb.approximation(lst, coef, x)
    np.testing.assert_almost_equal(a, b)


"### QFUNCTION ###"


def test_Qfunction_definition():
    """Test cython version of Q function."""
    env.reset()
    lst = cheb.get_index_list(2, 5)
    coef = [np.random.rand(len(lst)).tolist()]
    x = env.getstate()

    def dummyvaluefun(t, st, sotw):
        return 0

    q = qfunction.QFunction(env, 0, x, env.SOTW, dummyvaluefun)
    q.eval(np.array([0.2, 0.2]))

    assert True


"### OPTIMIZATION ###"


def xtest_LrGeoDice_optimization():
    """Test optimization of LrGeo-DICE."""

    with open("results/lrgeodice_deterministic.pickle", "rb") as f:
        res = pickle.load(f)
    initialguess = res.x
    initialguess[1::2] += 0.3
    res = modeltools.optimize(env, initialguess)
    print(res[0])
    assert res[0] > 9000
    with open("results/lrgeodice_det_modeltools_optimized.pickle", "wb") as f:
        pickle.dump(res, f)


"### DOMAIN EXPLORATION ###"


def xtest_domainGenerator():
    """Test domain generation."""
    with open("results/lrgeodice_deterministic.pickle", "rb") as f:
        res = pickle.load(f)
    initialguess = res.x
    initialguess[1::2] += 0.3

    domain_low, domain_high = modeltools.load_domain(env)
    for trial in range(100):
        traj = modeltools.sample_trajectory(env)
        step = 0
        for dl, st, dh in zip(domain_low, traj, domain_high):
            print(trial, step, "\n\n")
            step += 1
            for low, x, high in zip(dl, st, dh):
                print(low, x, high)
                assert x >= low
                assert x <= high
            print("--")


def test_nodeGenerator():
    """Test generation of Chebyshev nodes."""
    nodelist = []
    n_processors = 8
    ndim = 3
    n_approx_nodes = [5, 4, 7]

    # test distribution over processors
    for rank in range(n_processors):
        for node in cheb.node_generator(ndim, n_approx_nodes, rank, n_processors):
            print(node)
            assert nodelist.count(node.tolist()) == 0
            nodelist.append(node.tolist())
    assert len(nodelist) == np.prod(n_approx_nodes)

    # check value
    i = 0
    for a in nodelist[0]:
        print(a)
        np.testing.assert_almost_equal(a, -math.cos(math.pi / (2 * n_approx_nodes[i])))
        i += 1


def test_nodeOptimizer():
    """Test node optimization."""

    # we test the optimization of the last step in the optimal trajectory

    # get trajectory
    with open("results/lrgeodice_det_modeltools_optimized.pickle", "rb") as f:
        res = pickle.load(f)
    # res = modeltools.optimize(env)
    tmax = env.get_max_time()

    # find true value
    env.setstate(tmax - 1, res[2][tmax - 1], 0)
    _, v_true, _, _ = env.step(res[1][-2:])

    # get domain
    domain_low, domain_high = modeltools.load_domain(env)

    def finalvaluefun(t, st, sotw):
        return 0

    # estimate node value
    v_estimated, action_estimated = dptools.optimize_node(
        env, tmax - 1, res[2][tmax - 1], 0, res[1][-2:], finalvaluefun
    )
    print(res[1][-2:])
    print(action_estimated)
    # # estimate action value
    # _, action_estimated = dptools.optimize_node(env, tmax - 1, res[2][-1], 0, [0.3],
    #                                                 domain_low[tmax - 1],
    #                                                 domain_high[tmax - 1],
    #                                                 [[1, 2], [3, 4]], [[0, 0]])

    print(v_estimated, v_true)
    assert v_estimated >= v_true
    env.setstate(tmax - 1, res[2][tmax - 1], 0)
    _, v_observed, _, _ = env.step(action_estimated)
    np.testing.assert_almost_equal(v_estimated, v_observed)


"### Dynamic Programming algorithm ###"


def test_DPCoordinatorSetup():
    """Test DPC definition."""
    cheb_degree = 8
    n_cheb_nodes = 20
    dpc = coordinator.DPCoordinator(senv, cheb_degree, n_cheb_nodes)

    print(dpc.valuefun(0, env.getstate(), 0))
    assert dpc.valuefun(0, env.getstate(), 0) == 0


def xtest_recursion_step():
    """Test recursion step."""
    cheb_degree = 4
    n_cheb_nodes = 5
    dpc = coordinator.DPCoordinator(env, cheb_degree, n_cheb_nodes)
    t = env.get_max_time() - 1
    dpc.recursion_step(t)

    # get trajectory
    with open("results/lrgeodice_det_modeltools_optimized.pickle", "rb") as f:
        res = pickle.load(f)
    # res = modeltools.optimize(env)
    tmax = env.get_max_time()

    # find true value
    env.setstate(tmax - 1, res[2][-1], 0)
    _, v_true, _, _ = env.step(res[1][-2:])

    assert v_true >= dpc.valuefun(t, res[2][-1], 0)


def xtest_DPCoordinatorSaveLoad():
    """Test save-load mechanism DPC."""
    cheb_degree = 8
    n_cheb_nodes = 20
    dpc = coordinator.DPCoordinator(env, cheb_degree, n_cheb_nodes)
    t = env.get_max_time() - 1
    dpc.recursion_step(t)
    dpc.save("dpc1.dat")
    dpc2 = coordinator.load("dpc1.dat")

    res = modeltools.optimize(env)
    st = res[2][t]

    assert dpc2.valuefun(t, st, 0) == dpc.valuefun(t, st, 0)


def xtest_dpc_MSE():
    """Check that DPC is within error bounds."""
    cheb_degree = 8
    n_cheb_nodes = 20
    dpc = coordinator.DPCoordinator(env, cheb_degree, n_cheb_nodes)
    t = env.get_max_time() - 1
    dpc.recursion_step(t)
    mse = dpc.estimate_mse(t)
    print(mse)
    assert mse < 0.01


def xtest_DPC_timeEstimation():
    """Test time estimation DPC"""
    cheb_degree = 4
    n_cheb_nodes = 5

    dpc = coordinator.DPCoordinator(env, cheb_degree, n_cheb_nodes)

    estimated_duration = dpc.estimate_fit_time()

    print(estimated_duration)

    # t0 = timeit.default_timer()
    #
    # dpc.fit()
    #
    # t1 = timeit.default_timer()
    #
    # duration = t1 - t0
    # error_margin = .1
    # print('Estimation {} | Duration {}'.format(estimated_duration, duration))
    # assert duration < (1.1)*estimated_duration
    # assert duration > (0.4)*estimated_duration


"### STOCHASTIC OPTIMALISATION###"


def xtest_Economy_stochastic():
    """test stochasticity of economic model."""
    U = []

    for _ in range(3):
        senv.reset()
        u = 0
        for _ in range(senv.get_max_time()):
            _, r, _, _ = senv.step(0.3)
            u += r
        U.append(u)

    assert U[0] != U[1] or U[0] != U[2]


"### LR-GEODICE ###"


def xtest_DPCoordinatorSetup_withIAM():
    """Test DPC definition with LrGeo-DICE."""
    cheb_degree = 8
    n_cheb_nodes = 20
    dpc = coordinator.DPCoordinator(iam, cheb_degree, n_cheb_nodes)
    assert dpc.valuefun(0, env.getstate(), 0) == 0


def test_scenarios():
    """Test scenario dimensions"""
    # assert len(senv.scenario_probabilities()) == senv.get_number_of_worlds()
    assert len(env.scenario_probabilities()) == env.get_number_of_worlds()
