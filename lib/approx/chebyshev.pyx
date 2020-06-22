import math
cimport numpy as np
import numpy as np
import sys
import traceback
cimport chebyshev

# cpdef double simple_polynomial(int j, double x):
#    return math.cos(j*math.acos(x))


cpdef np.ndarray domain_to_box(np.ndarray x,
                               np.ndarray domain_low,
                               np.ndarray domain_high):
    return 2*(x - domain_low)/(domain_high - domain_low) - 1


cpdef np.ndarray box_to_domain(np.ndarray x,
                              np.ndarray domain_low,
                              np.ndarray domain_high):
    return (x + 1) / 2 * (domain_high - domain_low) + domain_low


cpdef double simple_polynomial(int j, double x):
    """Recursive formulation of chebyshev polynomial.

    This is slightly faster than the trigonometric formulation,
    and much faster than the implementation in numpy."""
    if x > 1 or x < -1:
      exc_type, exc_value, exc_traceback = sys.exc_info()
      print('here')
      traceback.print_tb(exc_traceback, limit=5)
      #sys.exit(1)
      raise ValueError('Argument should be in [-1, 1] but is {} instead'.format(x))

    if j == 0:
      return 1
    if j == 1:
      return x
    return 2*x*simple_polynomial(j-1, x) - simple_polynomial(j-2, x)


cpdef double ndim_polynomial(list J, np.ndarray X):
    cdef int j
    cdef double x
    cdef double res
    res = 1
    for j, x in zip(J,X):
      res = res * simple_polynomial(j, x)
    return res

cpdef double approximation(list J, list coef, np.ndarray x):
    cdef list j
    cdef double res
    cdef double c
    res = 0
    for c, j in zip(coef, J):
        res += c*ndim_polynomial(j, x)
    return res

cpdef get_index_list(dim, degree, x=None):
    """Generate all n-dimensional index vectors with max degree.

    Note that function generates vectors recursively.
    """
    if x is None:
        x = []
    nodes = []
    for i in range(degree+1):
        y = x[:]
        y.append(i)
        if sum(y) > degree:
            continue
        if dim == 1:
            nodes.append(y)
        else:
            nodes.extend(get_index_list(dim -1, degree, y))
    return nodes

def clean_index_list(degree, lst):
    lst_ = []
    degree = np.array(degree)
    for indexvec in lst:
        if np.all(indexvec < degree):
            lst_.append(indexvec)
    return lst_

def node_generator(int ndim, list m, int rank, int n_processes):
        """ Generate Chebyshev nodes in the [-1,1]^n plane (use in for-loop).

        Here we distribute the work over the different processes when the
        code is run in parallel.

        Args:
          ndim: number of dimensions
          m: list of number of nodes per dimension
          rank: identifier of current process (MPI)
          n_processes: total number of processes (MPI)
        """

        # loop only over the nodes for current process
        cdef int i, j, x
        cdef np.ndarray k, node
        for i in range(rank, np.prod(m), n_processes):

            # calculate the index in each individual dimension
            x = i
            k = np.zeros(ndim)
            for j in range(ndim):
                k[j] = x % m[j]
                x = (x - x % m[j]) / m[j]

            # calculate node
            node = np.zeros(ndim)
            for j in range(ndim):
                node[j] = -math.cos((2*k[j] + 1) * math.pi/(2 * m[j]))

            yield node
