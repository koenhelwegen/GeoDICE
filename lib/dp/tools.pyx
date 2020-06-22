cimport lib.dp.qfunction as qfunction
cimport numpy as np
import numpy as np
import scipy.optimize
cimport lib.approx.chebyshev as cheb
cimport tools


cpdef tuple optimize_node(env, int t, np.ndarray x, int sotw, np.ndarray a0,
                          valuefun):
    """Optimize node value."""

    # create q function & bounds
    q = qfunction.QFunction(env, t, x, sotw, valuefun)

    # optimize
    with q:
        bounds=env.get_action_bounds()
        for i in range(len(a0)):
          if a0[i] < bounds[i][0]:
            a0[i] = bounds[i][0]
          elif a0[i] > bounds[i][1]:
            a0[i] = bounds[i][1]
        res = scipy.optimize.minimize(q.eval, a0, bounds=bounds, method='slsqp', options={'ftol': 1e-7, 'maxiter': 1000})
        if a0[-1] != 0:
          a0 = res.x[:-1]
          res2 = scipy.optimize.minimize(q.eval, np.append(a0,0), bounds=bounds, method='slsqp', options={'ftol': 1e-7, 'maxiter': 1000})
          if res2.fun < res.fun:
            res = res2

    # check if optimization was successful
    if not res.success:
      print('WARNING: optimization failed (time {}, sotw {})'.format(t, sotw))

    return -res.fun, res.x
