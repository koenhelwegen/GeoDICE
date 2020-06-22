cimport numpy as np

cdef class QFunction(object):

  #cdef economy.Economy env
  cdef env
  cdef valuefun
  cdef np.ndarray x
  cdef int t
  cdef tuple img
  cdef list scenario_vector
  cdef int n_scenarios
  cdef int sotw

  cpdef double eval(self, np.ndarray action)
