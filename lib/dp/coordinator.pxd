cimport numpy as np

cdef class DPCoordinator(object):

    cdef public env
    cdef list domain_low, domain_high
    cdef list chebindlst, cheb_d
    cdef n_approx_nodes
    cdef int rank, n_processes, ndim, cheb_length, cheb_degree
    cdef int tmax
    cdef np.ndarray cheb_coef
    cdef int n_worlds

    cpdef double valuefun(self, int t, np.ndarray st, int sotw)
    cpdef tuple optimize_action_value(self, int t, np.ndarray st, int sotw, np.ndarray a0)
    cpdef void recursion_step(self, int t)
